import torch
import torch.nn as nn
import torch.nn.functional as F

def normalize_coords(coords_2D, batch_size, height, width):
    """
        Normalize 2D keypoint coordinates to lie in the range [-1, 1].

        Args:
            coords_2D (torch.tensor): 2D image coordinates store in order (u, v), (Bx2xN).
            batch_size (int): batch size.
            height (int): image height.
            width (int): image width.

        Returns:
            coords_2D_norm (torch.tensor): coordinates normalized to range [-1, 1] and stored in order (u, v), (BxNx2).
     """
    u_norm = (2 * coords_2D[:, 0, :].reshape(batch_size, -1) / (width - 1)) - 1
    v_norm = (2 * coords_2D[:, 1, :].reshape(batch_size, -1) / (height - 1)) - 1

    return torch.stack([u_norm, v_norm], dim=2)

def get_norm_descriptors(descriptor_map, sample=False, keypoints_norm=None):
    """
        Normalize the descriptors.

        Args:
            descriptor_map (torch.tensor): the descriptor map, either dense with one descriptor for each image pixel,
            BxCxHxW, or sparse with one descriptor for each keypoint, BxCxN.
            sample (bool): whether to sample descriptors for keypoints in the image or use descriptors for each
                           pixel.
            keypoints_norm (torch.tensor or None): normalized keypoint coordinates if we are sampling, otherwise None.

        Returns:
            descriptors_norm (torch.tensor): the normalized descriptors, (BxCx(N or HW))
    """

    if len(descriptor_map.size()) == 4:
        # The descriptor map has dense descriptors, one for each pixel.
        batch_size, channels, height, width = descriptor_map.size()

        if sample:
            # Sample descriptors for the given keypoints.
            descriptors = F.grid_sample(descriptor_map, keypoints_norm, mode='bilinear')     # BxCx1xN
            descriptors = descriptors.reshape(batch_size, channels, keypoints_norm.size(2))  # BxCxN
        else:
            descriptors = descriptor_map.reshape(batch_size, channels, height * width)  # BxCxHW
    else:
        # The descriptor map has sparse descriptors, one for each keypoint.
        descriptors = descriptor_map

    descriptors_mean = torch.mean(descriptors, dim=1, keepdim=True)           # Bx1x(N or HW)
    descriptors_std = torch.std(descriptors, dim=1, keepdim=True)             # Bx1x(N or HW)
    descriptors_norm = (descriptors - descriptors_mean) / descriptors_std     # BxCx(N or HW)

    return descriptors_norm

def get_scores(scores_map, keypoints_norm):
    """
        Sample scores for keypoints.

        Args:
            scores_map (torch.tensor): the scores for each pixel in the image, (Bx1xHxW).
            keypoints_norm (torch.tensor): normalized keypoint coordinates for sampling, (BxNx2).

        Returns:
            kpt_scores (torch.tensor): a score for each keypoint, (Bx1xN).
    """
    batch_size, _, num_points, _ = keypoints_norm.size()

    kpt_scores = F.grid_sample(scores_map, keypoints_norm, mode='bilinear')  # Bx1x1xN
    kpt_scores = kpt_scores.reshape(batch_size, 1, num_points)

    return kpt_scores

def get_keypoint_info(kpt_2D, scores_map, descriptors_map, disparity, stereo_cam):
    """
        Gather information we need associated with each detected keypoint. Compute the 3D point coordinates, determine
        which keypoints have valid 3D coordinates, get the normalized descriptor for each keypoint, and the score for
        each keypoint.

        Args:
            kpt_2D (torch.tensor): keypoint 2D image coordinates, (Bx2xN).
            scores_map (torch.tensor): scores for each pixel, (Bx1xHxW).
            descriptors_map (torch.tensor): descriptors for each pixel, (BxCxHxW).
            disparity (torch.tensor): disparity for one stereo pair, (BxHxW).
            stereo_cam (StereoCameraModel): stereo camera model.

        Returns:
            kpt_3D (torch.tensor): keypoint 3D coordinates in the sensor frame (left camera frame) given in
                                   homogeneous coordinates, (Bx4xN).
            valid (torch.tensor): 1 if the keypoint 3D coordinate is valid (i.e. falls in the accepted depth range) and
                                  0 otherwise, (Bx1xN).
            kpt_desc_norm (torch.tensor): Normalized descriptor for each keypoint, (BxCxN).
            kpt_scores (torch.tensor): score for each keypoint, (Bx1xN).

    """
    batch_size, _, height, width = scores_map.size()

    kpt_2D_norm = normalize_coords(kpt_2D, batch_size, height, width).unsqueeze(1)  # Bx1xNx2

    kpt_desc_norm = get_norm_descriptors(descriptors_map, True, kpt_2D_norm)

    kpt_scores = get_scores(scores_map, kpt_2D_norm)

    kpt_3D, valid = stereo_cam.inverse_camera_model(kpt_2D, disparity)

    return kpt_3D, valid, kpt_desc_norm, kpt_scores