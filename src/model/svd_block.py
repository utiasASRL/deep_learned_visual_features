""" Some of this code is based on Code from: https://github.com/WangYueFt/dcp/blob/master/model.py """

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.lie_algebra import se3_log, se3_inv

class SVDBlock(nn.Module):
    """
        Compute the relative pose between the source and target frames.
    """
    def __init__(self, T_s_v):
        super(SVDBlock, self).__init__()

        # Transform from vehicle to sensor frame.
        self.register_buffer('T_s_v', T_s_v)

    def forward(self, keypoints_3D_src, keypoints_3D_trg, weights):
        """
            Compute the pose, T_trg_src, from the source to the target frame.

            Args:
                keypoints_3D_src (torch,tensor, Bx4xN): 3D point coordinates of keypoints from source frame.
                keypoints_3D_trg (torch,tensor, Bx4xN): 3D point coordinates of keypoints from target frame.
                weights (torch.tensor, Bx1xN): weights in range (0, 1) associated with the matched source and target
                                               points.

            Returns:
                T_trg_src (torch.tensor, Bx4x4): relative transform from the source to the target frame.
        """
        batch_size, _, n_points = keypoints_3D_src.size()

        # Compute weighted centroids (elementwise multiplication/division)
        centroid_src = torch.sum(keypoints_3D_src[:, 0:3, :] * weights, dim=2, keepdim=True) / torch.sum(weights, dim=2,
                                                                                                         keepdim=True)  # Bx3x1
        centroid_trg = torch.sum(keypoints_3D_trg[:, 0:3, :] * weights, dim=2, keepdim=True) / torch.sum(weights, dim=2,
                                                                                                         keepdim=True)

        src_centered = keypoints_3D_src[:, 0:3, :] - centroid_src  # Bx3xN
        trg_centered = keypoints_3D_trg[:, 0:3, :] - centroid_trg

        W = torch.diag_embed(weights.reshape(batch_size, n_points))  # BxNxN
        w = torch.sum(weights, dim=2).unsqueeze(2)  # Bx1x1

        H = (1.0 / w) * torch.bmm(trg_centered, torch.bmm(W, src_centered.transpose(2, 1).contiguous()))  # Bx3x3

        U, S, V = torch.svd(H)

        det_UV = torch.det(U) * torch.det(V)
        ones = torch.ones(batch_size, 2).type_as(V)
        diag = torch.diag_embed(torch.cat((ones, det_UV.unsqueeze(1)), dim=1))  # Bx3x3

        # Compute rotation and translation (T_trg_src in sensor frame)
        R_trg_src = torch.bmm(U, torch.bmm(diag, V.transpose(2, 1).contiguous()))  # Bx3x3
        t_trg_src_insrc = centroid_src - torch.bmm(R_trg_src.transpose(2, 1).contiguous(), centroid_trg)  # Bx3x1
        t_src_trg_intrg = -R_trg_src.bmm(t_trg_src_insrc) # Translation from trg to src given in src frame.

        # Create translation matrix
        zeros = torch.zeros(batch_size, 1, 3).type_as(V)  # Bx1x3
        one = torch.ones(batch_size, 1, 1).type_as(V)  # Bx1x1
        trans_cols = torch.cat([t_src_trg_intrg, one], dim=1)  # Bx4x1
        rot_cols = torch.cat([R_trg_src, zeros], dim=1)  # Bx4x3
        T_trg_src = torch.cat([rot_cols, trans_cols], dim=2)  # Bx4x4

        # Convert from sensor to vehicle frame
        T_s_v = self.T_s_v.expand(batch_size, 4, 4)
        T_trg_src = se3_inv(T_s_v).bmm(T_trg_src).bmm(T_s_v)

        return T_trg_src