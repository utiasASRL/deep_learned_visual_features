import torch
import torch.nn as nn
import torch.nn.functional as F

class StereoCameraModel(nn.Module):
    """
        The stereo camera model.
    """

    def __init__(self, cu, cv, f, b):
        """
            Set up the stereo camera model with the stereo camera parameters.

            Args:
                cu (float): optical centre u coordinate.
                cv (float): optical centre v coordinate.
                f (float): focal length.
                b (float): stereo camera base line.
        """
        super(StereoCameraModel, self).__init__()
    
        self.cu = cu
        self.cv = cv
        self.f = f
        self.b = b
        
        # Matrices for camera model needed to projecting/reprojecting between the camera and image frames
        M, Q = self.set_camera_model_matrices(self.cu, self.cv, self.f, self.b)
        self.register_buffer('M', M)
        self.register_buffer('Q', Q)

    def set_camera_model_matrices(self, cu, cv, f, b):
        """
            Create the matrices needed to project 3D camera coordinates into 2D image coordinates (M) and compute
            3D camera coordinates from 2D image coordinates (Q).

            Args:
                cu (float): optical centre u coordinate.
                cv (float): optical centre v coordinate.
                f (float): focal length.
                b (float): stereo camera base line.

            Returns:
                M (torch.tensor): matrix used to project 3D camera coordinates to 2D image coordinates, (4x4).
                Q (torch.tensor): matrix used to compute 3D camera coordinates from 2D image coordinates, (4x4).
        """
        # Matrix needed to project 3D points into stereo camera coordinates.
        # [ul, vl, ur, vr]^T = (1/z) * M * [x, y, z, 1]^T (using left camera model)
        # 
        # [f, 0, cu,      0]
        # [0, f, cv,      0]
        # [f, 0, cu, -f * b]
        # [0, f, cv,      0]
        #
        M = torch.tensor([[self.f, 0.0, self.cu, 0.0], 
                          [0.0, self.f, self.cv, 0.0], 
                          [self.f, 0.0, self.cu, -(self.f * self.b)], 
                          [0.0, self.f, self.cv, 0.0]])
        
        # Matrix needed to transform image coordinates into 3D points.
        # [x, y, z, 1] = (1/d) * Q * [ul, vl, d, 1]^T
        # 
        # [b, 0, 0, -b * cu]
        # [0, b, 0, -b * cv]
        # [0, 0, 0,   f * b]
        # [0, 0, 1,       0]
        #
        Q = torch.tensor([[self.b, 0.0, 0.0, -(self.b * self.cu)], 
                          [0.0, self.b, 0.0, -(self.b * self.cv)], 
                          [0.0, 0.0, 0.0, self.f * self.b], 
                          [0.0, 0.0, 1.0, 0.0]])

        return M, Q

    def normalize_coords(self, coords_2D, batch_size, height, width):
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

    def check_valid_disparity(self, disparity):
        """
            Check if disparity falls withing our chosen accepted range (0.1m - 400.0m).

            Args:
                disparity (torch.tensor): the disparity map for the image, (Bx1xN).

            Returns:
                valid_disparity (torch.tensor): True for valid disparity values, False otherwise.
        """
        disp_min = (self.f * self.b) / 400.0  # Farthest point 400 m
        disp_max = (self.f * self.b) / 0.1    # Closest point 0.1 m

        return (disparity >= disp_min) & (disparity <= disp_max)

    def camera_to_image(self, cam_coords, M):
        """
            Project 3D points given in the camera frame into 2D image coordinates.

            Args:
                cam_coords (torch.tensor): 3D camera coordinates given as homogeneous coordinates, (Bx4xN).
                M (torch.tensor): matrix for projecting points into image, (Bx4x4).

            Returns:
                img_coords (torch.tensor): 2D image coordinates given in order (ul, vl, ur, vr), (Bx4xN).
        """
        batch_size, _, num_points = cam_coords.size()
        
        # [Ul, Vl, Ur, Vr] = M * [x, y, z, 1]^T
        img_coords = M.bmm(cam_coords)

        inv_z = 1.0 / (cam_coords[:, 2, :].reshape(batch_size, 1, num_points))  # Bx1xN, elementwise division.
        img_coords = img_coords * inv_z                                         # Bx4xN, elementwise multiplication.

        return img_coords

    def camera_model(self, cam_coords):
        """
            Project 3D points given in the camera frame into image coordinates.

            Args:
                cam_coords (torch.tensor): 3D camera coordinates given as homogeneous coordinates, (Bx4xN).

            Returns:
                img_coords (torch.tensor): 2D image coordinates given in order (ul, vl, ur, vr), (Bx4xN).
        """
        batch_size = cam_coords.size(0)

        # Expand fixed matrix to the correct batch size.
        M = self.M.expand(batch_size, 4, 4).cuda()

        # Get the image coordinates.
        img_coords = self.camera_to_image(cam_coords, M)

        if (torch.sum(torch.isnan(img_coords)) > 0) or (torch.sum(torch.isinf(img_coords)) > 0):
            print('Warning: Nan or Inf values in image coordinate tensor.')
            raise ValueError("Nan or Inf in image coordinates")

        return img_coords

    def image_to_camera(self, img_coords, disparity, Q):
        """
            Compute 3D point coordinates (in the camera frame) for the given 2D image coordinates.

            Args:
                img_coords (torch.tensor): 2D image coordinates given in order (u, v), (Bx2xN).
                disparity (torch.tensor): the disparity map for the image, (BxHxW).
                Q (torch.tensor): matrix for reprojecting image coordinates, (Bx4x4).

            Returns:
                cam_coords (torch.tensor): 3D points in camera frame given as homogeneous coordinates, (Bx4xN).
                valid_points (torch.tensor): value 1 for each valid 3D point with depth in accepted range, otherwise 0,
                                             (Bx1xN).
        """
        batch_size, height, width = disparity.size()
        disparity = disparity.unsqueeze(1)
        num_points = img_coords.size(2)

        if (torch.sum(disparity == 0.0) > 0):
            print('Warning: 0.0 in disparities.')

        # Sample disparity for the image coordinates.
        img_coords_norm = self.normalize_coords(img_coords, batch_size, height, width).unsqueeze(1)  # Bx1xNx2

        point_disparities = F.grid_sample(disparity, img_coords_norm, mode='nearest', padding_mode='border')  # Bx1x1xN
        point_disparities = point_disparities.reshape(batch_size, 1, num_points)  # Bxx1xN
        valid_points = self.check_valid_disparity(point_disparities)  # Bx1xN

        # Create the [ul, vl, d, 1] vector
        ones = torch.ones(batch_size, num_points).type_as(disparity)
        uvd1_pixel_coords = torch.stack((img_coords[:, 0, :],
                                         img_coords[:, 1, :],
                                         point_disparities[:, 0, :],
                                         ones),
                                        dim=1)  # Bx4xN

        # [X, Y, Z, d]^T = Q * [ul, vl, d, 1]^T
        cam_coords = Q.bmm(uvd1_pixel_coords)  # Bx4xN

        # [x, y, z, 1]^T = (1/d) * [X, Y, Z, d]^T
        inv_disparity = (1.0 / point_disparities)  # Elementwise division
        cam_coords = cam_coords * inv_disparity    # Elementwise multiplication

        return cam_coords, valid_points

    def inverse_camera_model(self, img_coords, disparity):
        """
            Compute 3D point coordinates (in the camera frame) from the given 2D image coordinates.

            Args:
                img_coords (torch.tensor): 2D image coordinates given in order (u, v), (Bx2xN).
                disparity (torch.tensor): the disparity map for the image, (BxHxW).

            Returns:
                cam_coords (torch.tensor): 3D points in camera frame given as homogeneous coordinates, (Bx4xN).
                valid_points (torch.tensor): value 1 for each valid 3D point with depth in accepted range, otherwise 0,
                                             (Bx1xN).
        """
        batch_size, height, width = disparity.size()
        
        # Expand fixed matrix to the correct batch size
        Q = self.Q.expand(batch_size, 4, 4).cuda()

        # Get the 3D camera coordinates.
        cam_coords, valid_points = self.image_to_camera(img_coords, disparity, Q)

        if (torch.sum(torch.isnan(cam_coords)) > 0) or (torch.sum(torch.isinf(cam_coords)) > 0):
            print('Warning: Nan or Inf values in camera coordinate tensor.')
            raise ValueError("Nan or Inf in camera coordinates")

        return cam_coords, valid_points
            
       

