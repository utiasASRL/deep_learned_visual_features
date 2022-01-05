import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightBlock(nn.Module):
    """
        Computes the weights for matched points pairs that will be used when computing the pose.
    """
    def __init__(self):
        super(WeightBlock, self).__init__()

    def forward(self, kpt_desc_norm_src, kpt_desc_norm_pseudo, kpt_scores_src, kpt_scores_pseudo):
        """
            Compute the weights for matched point pairs.

            Args:
                kpt_desc_norm_src (torch.tensor, BxCxN): normalized descriptors for the source keypoints.
                kpt_desc_norm_pseudo (torch.tensor, BxCxN): normalized descriptors for the target pseudo keypoints.
                kpt_scores_src (torch.tensor, Bx1xN): scores for the source keypoints.
                kpt_scores_pseudo (torch.tensor, Bx1xN): scores for the target pseudo keypoints.

            Returns:
                weights (torch.tensor, Bx1xN): weights for each matched point pair in range [0, 1] to be used when
                                               computing the pose.
        """
        batch_size, channels, n_points = kpt_desc_norm_src.size()

        # Get the zncc between each matched point pair
        desc_src = kpt_desc_norm_src.transpose(2, 1).reshape(batch_size * n_points, channels)        # BNxC
        desc_pseudo = kpt_desc_norm_pseudo.transpose(2, 1).reshape(batch_size * n_points, channels)  # BNxC

        desc_match_val = torch.matmul(desc_src.unsqueeze(1), desc_pseudo.unsqueeze(2)) / desc_pseudo.size(1) # BNx1

        desc_match_val = desc_match_val.reshape(batch_size, 1, n_points) + 1.0  # Range [0, 2] Bx1xN

        weights = 0.5 * desc_match_val * kpt_scores_src * kpt_scores_pseudo     # Range [0, 1]

        return weights