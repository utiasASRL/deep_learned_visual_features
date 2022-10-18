'''A script that shows how to pass an image to the network to get keypoints, descriptors and scrores. '''

import sys
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.unet import UNet
from src.model.keypoint_block import KeypointBlock
from src.utils.keypoint_tools import normalize_coords, get_norm_descriptors, get_scores


def get_keypoint_info(kpt_2D, scores_map, descriptors_map):
    """
        Gather information we need associated with each detected keypoint. Compute the normalized 
        descriptor and the score for each keypoint.

        Args:
            kpt_2D (torch.tensor): keypoint 2D image coordinates, (Bx2xN).
            scores_map (torch.tensor): scores for each pixel, (Bx1xHxW).
            descriptors_map (torch.tensor): descriptors for each pixel, (BxCxHxW).

        Returns:
            kpt_desc_norm (torch.tensor): Normalized descriptor for each keypoint, (BxCxN).
            kpt_scores (torch.tensor): score for each keypoint, (Bx1xN).

    """
    batch_size, _, height, width = scores_map.size()

    kpt_2D_norm = normalize_coords(kpt_2D, batch_size, height, width).unsqueeze(1)  # Bx1xNx2

    kpt_desc_norm = get_norm_descriptors(descriptors_map, True, kpt_2D_norm)

    kpt_scores = get_scores(scores_map, kpt_2D_norm)

    return kpt_desc_norm, kpt_scores


class LearnedFeatureDetector(nn.Module):
    """ 
        Class to detect learned features.
    """
    def __init__(self, n_channels, layer_size, window_height, window_width, image_height, image_width, checkpoint_path, cuda):
        """
            Set the variables needed to initialize the network.

            Args:
                num_channels (int): number of channels in the input image (we use 3 for one RGB image).
                layer_size (int): size of the first layer if the encoder. The size of the following layers are
                                  determined from this.
                window_height (int): height of window, inside which we detect one keypoint.
                window_width (int): width of window, inside which we detect one keypoint.
                image_height (int): height of the image.
                image_width (int): width of the image.
                checkpoint_path (string): path to where the network weights are stored.
                cuda (bool): true if using the GPU.
        """
        super(LearnedFeatureDetector, self).__init__()

        self.cuda = cuda
        self.n_classes = 1
        self.n_channels = n_channels
        self.layer_size = layer_size
        self.window_h = window_height
        self.window_w = window_width
        self.height = image_height
        self.width = image_width

        # Load the network weights from a checkpoint.
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
        else:
            raise RuntimeError(f'The specified checkpoint path does not exists: {checkpoint_path}')

        self.net = UNet(self.n_channels, self.n_classes, self.layer_size)
        # self.net = UNet(self.n_channels, self.n_classes, self.layer_size, self.height, self.width, checkpoint)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.keypoint_block = KeypointBlock(self.window_h, self.window_w, self.height, self.width)
        self.sigmoid = nn.Sigmoid()

        if cuda:
            self.net.cuda()
            self.keypoint_block.cuda()

        self.net.eval()

    def run(self, image_tensor):
        """
            Forward pass of network to get keypoint detector values, descriptors and, scores

            Args:
                image_tensor (torch.tensor, Bx3xHxW): RGB images to input to the network.

            Returns:
                keypoints (torch.tensor, Bx2xN): the detected keypoints, N=number of keypoints.
                descriptors (torch.tensor, BxCxN): descriptors for each keypoint, C=496 is length of descriptor.
                scores (torch.tensor, Bx1xN): an importance score for each keypoint.

        """
        if self.cuda:
            image_tensor = image_tensor.cuda()

        detector_scores, scores, descriptors = self.net(image_tensor)
        scores = self.sigmoid(scores)

        # Get 2D keypoint coordinates from detector scores, Bx2xN
        keypoints = self.keypoint_block(detector_scores)

        # Get one descriptor and scrore per keypoint, BxCxN, Bx1xN, C=496.
        point_descriptors_norm, point_scores = get_keypoint_info(keypoints, scores, descriptors)

        return keypoints.detach().cpu(), point_descriptors_norm.detach().cpu(), point_scores.detach().cpu()


if __name__ == '__main__':

    cuda = False
    checkpoint = '/home/mona/data/deep_learned_features/networks/network_multiseason_inthedark_layer16.pth'
    learned_feature_detector = LearnedFeatureDetector(n_channels=3, 
                                                      layer_size=16, 
                                                      window_height=16, 
                                                      window_width=16, 
                                                      image_height=384, 
                                                      image_width=512,
                                                      checkpoint_path=checkpoint,
                                                      cuda=cuda)

    
    test_image = torch.rand(1, 3, 384, 512)
    keypoints, descriptors, scores = learned_feature_detector.run(test_image)
    
    print(keypoints.size())
    print(descriptors.size())
    print(scores.size())