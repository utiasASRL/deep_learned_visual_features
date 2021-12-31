import torch
import torch.nn as nn
import torch.nn.functional as F

class MatcherBlock(nn.Module):
    """
        Match the descriptors corresponding to the source keypoints to all descriptors in the target frame. Use the
        values from descriptor matching to compute the coordinates of the matched keypoints in the target frame, which
        we call pseudo-points.
    """
    def __init__(self, window_height, window_width):
        super(MatcherBlock, self).__init__()

        self.window_height = window_height
        self.window_width = window_width
        self.temperature = 0.01

    def forward(self, kpt_2D_src, kpt_2D_trg, kpt_desc_norm_src, kpt_desc_norm_trg):
        """
            Compute coordinates for the matches target keypoints, which we refer to as pseudo-points.

            Args:
                kpt_2D_src (torch.tensor, Bx2xN): sparse keypoints detected in the source image.
                kpt_2D_trg (torch.tensor, Bx2x(N or HW)): coordinates of target keypoints to match. If we are doing
                                                          dense matching these will be the coordinates of all the pixels
                                                          in the target image (Bx2xWH). If we are doing sparse matching,
                                                          they will be the coordinates of the sparse detected keypoints
                                                          in the target image (Bx2xN).
                kpt_desc_norm_src (torch.tensor, BxCxN): Normalized descriptors corresponding to the source keypoints.
                                                         (C is the length of the descriptor).
                kpt_desc_norm_trg (torch.tensor, BxCx(N or HW)): Normalized descriptors corresponding to the target
                                                                 keypoints. BxCxHW for dense matching and BxCxN for
                                                                 sparse matching.

            Returns:
                kpt_2D_pseudo (torch.tensor, Bx2xN): keypoint coordinates of the matched pseudo-points in the target
                                                     frame.
         """

        batch_size, _, n_points = kpt_2D_src.size()

        # For all source keypoints, compute match value between each source point and all points in target frame.
        # Apply softmax for each source point along the dimension of the target points.
        match_vals = torch.matmul(kpt_desc_norm_src.transpose(2, 1), kpt_desc_norm_trg) \
                     / float(kpt_desc_norm_trg.size(1))        # BxNx(HW or N)

        match_vals = (match_vals + 1.0) # [-1, 1] -> [0, 2]
        soft_match_vals = F.softmax(match_vals / self.temperature, dim=2)  # BxNx(HW or N)

        # Compute pseudo-point as weighted sum of point coordinates from target image, Bx2xN
        kpt_2D_pseudo = torch.matmul(kpt_2D_trg, soft_match_vals.transpose(2, 1))

        return kpt_2D_pseudo

