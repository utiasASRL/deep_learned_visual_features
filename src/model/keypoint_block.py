
import torch
import torch.nn as nn
import torch.nn.functional as FS

class KeypointBlock(nn.Module):
    """
        Given the dense detector values from the UNet decoder, compute the 2D keypoints.
    """
    def __init__(self, window_height, window_width, image_height, image_width):
        super(KeypointBlock, self).__init__()
        self.window_height = window_height
        self.window_width = window_width
        self.temperature = 1.0

        v_coords, u_coords = torch.meshgrid([torch.arange(0, image_height), torch.arange(0, image_width)])
        v_coords = v_coords.unsqueeze(0).float()  # 1 x H x W
        u_coords = u_coords.unsqueeze(0).float()

        self.register_buffer('v_coords', v_coords)
        self.register_buffer('u_coords', u_coords)

        self.sigmoid = nn.Sigmoid()

    def forward(self, detector_values):
        """
            Given a tensor of detector values (same width/height as the original image), divide the tensor into
            windows and use a spatial softmax over each window. The 2D coordinates of one keypoint is estimated for each
            window.

            Args:
                detector_values (torch.tensor, Bx1xHxW): Tensor of detector values from the network decoder.
            Returns:
                keypoints_2D (torch.tensor, Bx2xN): Keypoint coordinates.
        """

        batch_size, _, height, width = detector_values.size()

        v_windows = F.unfold(self.v_coords.expand(batch_size, 1, height, width),
                             kernel_size=(self.window_height, self.window_width),
                             stride=(self.window_h, self.window_w))  # B x n_window_elements x n_windows
        u_windows = F.unfold(self.u_coords.expand(batch_size, 1, height, width),
                             kernel_size=(self.window_height, self.window_width),
                             stride=(self.window_height, self.window_width))

        detector_values_windows = F.unfold(detector_values,
                                           kernel_size=(self.window_height, self.window_width),
                                           stride=(self.window_height, self.window_width))  # B x n_wind_elements x n_windows

        softmax_attention = F.softmax(detector_values_windows / self.temperature, dim=1)  # B x n_wind_elements x n_windows

        expected_v = torch.sum(v_windows * softmax_attention, dim=1)  # B x n_windows
        expected_u = torch.sum(u_windows * softmax_attention, dim=1)
        keypoints_2D = torch.stack([expected_u, expected_v], dim=2).transpose(2, 1)  # B x 2 x n_windows

        return keypoints_2D
