import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.svd_block import SVDBlock
from src.utils.lie_algebra import se3_log, se3_inv
from src.utils.stereo_camera_model import StereoCameraModel

class RANSACBlock(nn.Module):
    """
        Use RANSAC for outlier rejection during inference.
    """

    def __init__(self, config, T_s_v):
        """
            Initialize RANSAC by setting values used to determine how many iterations to run.

            Args:
                config (dict): configuration parameters.
                T_s_v (torch.tensor): 4x4 transformation matrix providing the transform from the vehicle frame to the
                                      sensor frame.
        """
        super(RANSACBlock, self).__init__()

        # Transform from vehicle to sensor frame.
        self.register_buffer('T_s_v', T_s_v)

        self.stereo_cam = StereoCameraModel(config['stereo']['cu'], config['stereo']['cv'],
                                            config['stereo']['f'], config['stereo']['b'])

        # The minimum number of inliers we want, quit RANSAC when this is achieved.
        self.inlier_threshold = config['outlier_rejection']['inlier_threshold']

        # Error must be smaller than or equal to threshold to this be considered inlier.
        dim_key = config['outlier_rejection']['dim'][0]
        self.error_tolerance = config['outlier_rejection']['error_tolerance'][dim_key]

        # Maximum number of iterations to run before giving up.
        self.num_iterations =  config['outlier_rejection']['num_iterations']

        # SVD is used to estimate pose.
        self.svd = SVDBlock(T_s_v)

    def forward(self, keypoints_3D_src, keypoints_3D_trg, keypoints_2D_trg, valid_pts_src, valid_pts_trg, weights, dim):
        """
            Outlier rejection with RANSAC.

            Args:
                keypoints_3D_src (torch,tensor, Bx4xN): 3D point coordinates of keypoints from source frame.
                keypoints_3D_trg (torch,tensor, Bx4xN): 3D point coordinates of keypoints from target frame.
                keypoints_2D_trg (torch,tensor, Bx2xN): 2D image coordinates of keypoints from source frame.
                valid_pts_src (torch.tensor, Bx1xN): Values (0 or 1) to indicate if a keypoint in source frame is valid
                                                     (i.e. can be used for pose computation).
                valid_pts_trg (torch.tensor, Bx1xN): Values (0 or 1) to indicate if a keypoint in target frame is valid
                                                     (i.e. can be used for pose computation).
                weights (torch.tensor, Bx1xN): weights in range (0, 1) associated with the matched source and
                                               target points.
                dim (str):  '2D' or '3D' to specify if error should be taken between 2D image coordinates or 3D point
                            coordinates.

            Returns:
                inliers (torch.tensor, BxN):
        """
        batch_size, _, n_points = keypoints_3D_src.size()

        valid = valid_pts_src & valid_pts_trg
        weights_svd = copy.deepcopy(weights)
        weights_svd[valid == 0] = 0.0

        pts_3D_src = keypoints_3D_src.detach()
        pts_3D_trg = keypoints_3D_trg.detach()
        pts_2D_trg = keypoints_2D_trg.detach()

        max_num_inliers = torch.zeros(batch_size).type_as(pts_3D_src) # Keep track of highest number of inliers so far.
        inliers = torch.zeros(batch_size, n_points, dtype=torch.bool).cuda()

        i = 0
        ransac_complete = torch.zeros(batch_size).type_as(pts_3D_src).int()

        while (i < self.num_iterations) and (torch.sum(ransac_complete) < batch_size):

            # Pick a random subset of 6 point pairs (3 sufficient, but some points will have weight 0 so pick a
            # few more than needed to increase probability of getting points with rank 3).
            rand_index = torch.randint(0, n_points, size=(batch_size, 6)).type_as(pts_3D_src).long()
            rand_index = rand_index.unsqueeze(1)
            rand_pts_3D_src = torch.gather(pts_3D_src, dim=2, index=rand_index.expand(batch_size, 4, 6))  # 1x4xM
            rand_pts_3D_trg = torch.gather(pts_3D_trg, dim=2, index=rand_index.expand(batch_size, 4, 6))  # 1x4xM
            rand_weights = torch.gather(weights.detach(), dim=2, index=rand_index)  # 1x1xM

            # Run SVD
            try:
                T_trg_src = self.svd(rand_pts_3D_src, rand_pts_3D_trg, rand_weights) # pose in vehicle frame
            except RuntimeError as e:
                print(e)
                print("RANSAC SVD did not converge, re-doing iteration {}".format(i))
                print("weights: {}".format(rand_weights[0, 0, :]))
                print("rank src pts: {}".format(torch.matrix_rank(rand_pts_3D_src[0, 0:3, :])))
                print("rank trg pts: {}".format(torch.matrix_rank(rand_pts_3D_trg[0, 0:3, :])), flush=True)
                continue

            # Find number of inliers
            T_s_v = self.T_s_v.expand(batch_size, 4, 4)
            T_trg_src_cam = T_s_v.bmm(T_trg_src).bmm(se3_inv(T_s_v)) # pose in camera frame
            pts_3D_trg_est = T_trg_src_cam.bmm(pts_3D_src)
            pts_2D_trg_est = self.stereo_cam.camera_model(pts_3D_trg_est)[:, 0:2, :]

            if dim == '2D':
                err_pts = torch.norm(pts_2D_trg - pts_2D_trg_est, dim=1)  # BxN
            else:
                err_pts = torch.norm(pts_3D_trg - pts_3D_trg_est, dim=1)  # BxN

            err_pts_small = err_pts < self.error_tolerance
            err_pts_small[valid_pts_src[:, 0, :] == 0] = 0
            err_pts_small[valid_pts_trg[:, 0, :] == 0] = 0

            num_inliers = torch.sum(err_pts_small, dim=1)

            fraction_inliers = num_inliers.float() / n_points
            enough_inliers = fraction_inliers > self.inlier_threshold
            ransac_complete = ransac_complete | enough_inliers

            for b in range(batch_size):
                if num_inliers[b] > max_num_inliers[b]:
                    max_num_inliers[b] = num_inliers[b]
                    inliers[b, :] = err_pts_small[b, :]

            i += 1

        return inliers