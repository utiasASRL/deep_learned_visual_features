import torch
import torch.nn as nn
import torch.nn.functional as F

from src.networks.keypoint_block import KeypointBlock
from src.networks.matcher_block import MatcherBlock
from src.networks.ransac_block import RANSACBlock
from src.networks.svd_block import SVDBlock
from src.networks.unet import UNet, UNetLarge, UNetVGG16
from src.networks.weight_block import SVDWeightBlock
from src.utils.keypoint_tools import get_keypoint_info, get_norm_descriptors, normalize_coords
from src.utils.lie_algebra import se3_inv, se3_log, se3_exp
from src.utils.stereo_camera_model import StereoCameraModel

class Pipeline(nn.Module):
    def __init__(self, config):
        super(Pipeline, self).__init__()

        self.config = config

        self.window_h = config['network']['window_h']
        self.window_w = config['network']['window_w']
        self.dense_matching = config['network']['dense_matching']

        # Image height and width
        self.height = config['dataset']['height']
        self.width = config['dataset']['width']

        # Transform from sensor to vehicle.
        T_s_v = torch.tensor([[0.000796327, -1.0, 0.0, 0.119873],
                              [-0.330472, -0.000263164, -0.943816, 1.49473],
                              [0.943815, 0.000751586, -0.330472, 0.354804],
                              [0.0, 0.0, 0.0, 1.0]])
        self.register_buffer('T_s_v', T_s_v)

        # Set up the different blocks needed in the pipeline
        self.keypoint_block = KeypointBlock(self.window_h, self.window_w, self.height, self.width)
        self.matcher_block = MatcherBlock(self.window_h, self.window_w)
        self.weight_block = WeightBlock()
        self.svd_block = SVDBlock(self.T_s_v)

        dim_key = self.config['outlier_rejection']['dim'][0]
        self.ransac_block = RANSACBlock(config['outlier_rejection']['inlier_threshold'],
                                        config['outlier_rejection']['error_tolerance'][dim_key],
                                        config['outlier_rejection']['num_iterations'],
                                        self.T_s_v)

        self.stereo_cam = StereoCameraModel(config['stereo']['cu'], config['stereo']['cv'],
                                            config['stereo']['f'], config['stereo']['b'])

        # Set up the coordinates for the image pixels, only need to do this once.
        v_coord, u_coord = torch.meshgrid([torch.arange(0, self.height), torch.arange(0, self.width)])
        v_coord = v_coord.reshape(self.height * self.width).float()  # HW
        u_coord = u_coord.reshape(self.height * self.width).float()
        image_coords = torch.stack((u_coord, v_coord), dim=0)  # 2xHW
        self.register_buffer('image_coords', image_coords)

    def forward(self, net, images, disparities, poses_se3, poses_log, epoch):

        ################################################################################################################
        #  Setup of variables
        ################################################################################################################

        batch_size = images.size(0)

        im_channels = [0, 1, 2, 6, 7, 8] # Picking out the left images from the four stereo src and trg images.
        images = images[:, im_channels, :, :].cuda()
        disparities = disparities.cuda()

        for i in range(len(poses_se3)):
            poses_se3[i] = poses_se3[i].cuda() # Pose 4x4 matrices
            poses_log[i] = poses_log[i].cuda() # Pose 6x1 vectors

        # Variables to store the loss
        losses = {'total': torch.zeros(1).cuda()}
        mse_loss_fn = torch.nn.MSELoss()
        loss_weights = self.config['loss_weights']


        ################################################################################################################
        #  Get keypoints and associated info for the source and target frames
        ################################################################################################################

        # Pass the images through the UNet (Bx1xHxW, Bx1xHxW, BxCxHxW)
        detector_scores_src, scores_src, descriptors_src = net(images[:, :3, :, :])
        detector_scores_trg, scores_trg, descriptors_trg = net(images[:, 3:, :, :])

        # Get 2D keypoint coordinates from detector scores, Bx2xN
        kpt_2D_src = self.keypoint_block(detector_scores_src)
        kpt_2D_trg = self.keypoint_block(detector_scores_trg)

        # Get 3D point coordinates, normalized descriptors, and scores associated with each individual keypoint
        # (Bx4xN, BxCxN, Bx1xN).
        kpt_3D_src, kpt_valid_src, kpt_desc_norm_src, kpt_scores_src = get_keypoint_info(kpt_2D_src,
                                                                                         scores_src,
                                                                                         descriptors_src,
                                                                                         disparities[:, 0, :, :],
                                                                                         self.stereo_cam,
                                                                                         'init')

        kpt_3D_trg, kpt_valid_trg, kpt_desc_norm_trg, kpt_scores_trg = get_keypoint_info(kpt_2D_trg,
                                                                                         scores_trg,
                                                                                         descriptors_trg,
                                                                                         disparities[:, 0, :, :],
                                                                                         self.stereo_cam,
                                                                                         'init')


        ################################################################################################################
        # Match keypoints from the source and target frames
        ################################################################################################################

        if self.config['network']['args']['dense_matching']:

            # Match against descriptors for each pixel in the target.
            desc_norm_trg_dense = get_norm_descriptors(descriptors_trg)
            kpt_2D_trg_dense = self.image_coords.unsqueeze(0).expand(batch_size, 2, self.height * self.width)

            # Compute the coordinates of the matched keypoints in the target frame, which we refer to as pseudo points.
            kpt_2D_pseudo = self.matcher_block(kpt_2D_src, kpt_2D_trg_dense, kpt_desc_norm_src, desc_norm_trg_dense)

        else:

            # Match only against descriptors associated with detected keypoints in the target frame.
            kpt_2D_pseudo = self.matcher_block(kpt_2D_src, kpt_2D_trg, kpt_desc_norm_src, kpt_desc_norm_trg)

        # Get 3D point coordinates, normalized descriptors, and scores associated with each individual matched pseudo
        # point in the target frame (Bx4xN, BxCxN, Bx1xN).
        kpt_3D_pseudo, kpt_valid_pseudo, kpt_desc_norm_pseudo, kpt_scores_pseudo = get_keypoint_info(kpt_2D_pseudo,
                                                                                                   scores_trg,
                                                                                                   descriptors_trg,
                                                                                                   disparities[:,1,:,:],
                                                                                                   self.stereo_cam,
                                                                                                   'pseudo')

        # Compute the weight associated with each matched point pair. They will be used when computing the pose.
        weights = self.weight_block(kpt_desc_norm_src, kpt_desc_norm_pseudo, kpt_scores_src, kpt_scores_pseudo)


        ################################################################################################################
        # Outlier rejection
        ################################################################################################################

        # Find the inliers either by using the ground truth pose (training) or RANSAC (inference).
        valid_inliers = torch.ones(kpt_valid_src.size()).type_as(kpt_valid_src)

        if self.config['outlier_rejection']['on'] and (self.config['outlier_rejection']['type'] == 'ground_truth'):

            # For training, we use the ground truth pose to find the ground truth location of the target points by
            # transforming them from the source frame to the target frames.

            if 'plane' in self.config['outlier_rejection']['dim']:
                # If we use only part of the ground truth pose (x, y, heading) we transform the points in the plane only.
                log_pose = poses_log[0]

                one = torch.ones(batch_size).cuda()    # B
                zero = torch.zeros(batch_size).cuda()  # B

                rot_col1 = torch.stack([torch.cos(log_pose[:, 5]), torch.sin(log_pose[:, 5]),
                                            zero.clone(), zero.clone()], dim=1)                             # Bx4
                rot_col2 = torch.stack([-torch.sin(log_pose[:, 5]), torch.cos(log_pose[:, 5]),
                                            zero.clone(), zero.clone()], dim=1)                             # Bx4
                rot_col3 = torch.stack([zero.clone(), zero.clone(), one.clone(), zero.clone()], dim=1)      # Bx4
                trans_col = torch.stack([log_pose[:, 0], log_pose[:, 1], zero.clone(), one.clone()], dim=1) # Bx4

                T_trg_src_inl_gt = torch.stack([rot_col1, rot_col2, rot_col3, trans_col], dim=2)  # Bx4x4

            else:
                # We use the full ground truth pose to transform the points.
                T_trg_src_inl_gt = poses[v_pair]

            T_s_v = self.T_s_v.expand(batch_size, 4, 4)
            T_trg_src_inl_gt_sensor = T_s_v.bmm(T_trg_src_inl_gt).bmm(se3_inv(T_s_v)) # Transform pose to sensor frame.

            # The ground truth target points, which will be compared to the matched target points.
            kpt_3D_pseudo_gt = T_trg_src_inl_gt_sensor.bmm(kpt_3D_src)                    # Bx4xN
            kpt_2D_pseudo_gt = self.stereo_cam.camera_model(kpt_3D_pseudo_gt)[:, 0:2, :]  # Bx2xN

            # Find the residual between the ground truth and matched point coordinates and determine the inliers.
            if '3D' in self.config['outlier_rejection']['dim']:
                err = torch.norm(kpt_3D_pseudo - kpt_3D_pseudo_gt, dim=1)  # B x N
                inliers_3D = err < self.config['outlier_rejection']['error_tolerance']['3D']
                valid_inliers = valid_inliers & (inliers_3D.unsqueeze(1))

            if ('2D' in self.config['outlier_rejection']['dim']):
                err = torch.norm(kpt_2D_pseudo - kpt_2D_pseudo_gt, dim=1)  # B x N
                inliers_2D = err < self.config['outlier_rejection']['error_tolerance']['2D']
                valid_inliers = valid_inliers & (inliers_2D.unsqueeze(1))

            if 'plane' in self.config['outlier_rejection']['dim']:
                # Do comparison in the vehicle frame, not the sensor frame.
                kpt_3D_pseudo_gt_vehicle = T_trg_src_inl_gt.bmm(se3_inv(T_s_v).bmm(kpt_3D_src))  # Bx4xN
                kpt_3D_pseudo_vehicle = se3_inv(T_s_v).bmm(kpt_3D_pseudo)
                err = torch.norm(kpt_3D_pseudo_vehicle[:, 0:2, :] - kpt_3D_pseudo_gt_vehicle[:, 0:2, :], dim=1)  # BxN
                inliers_plane = err < self.config['outlier_rejection']['error_tolerance']['plane']
                valid_inliers = valid_inliers & (inliers_plane.unsqueeze(1))

        if self.config['outlier_rejection']['on'] and (self.config['outlier_rejection']['type'] == 'ransac'):
            # Find outlier based on RANSAC.
            ransac_inliers = self.ransac_block(kpt_3D_src,
                                               kpt_3D_pseudo,
                                               kpt_2D_pseudo,
                                               kpt_valid_src,
                                               kpt_valid_pseudo,
                                               weights,
                                               self.config['outlier_rejection']['dim'][0])

            valid_inliers = ransac_inliers.unsqueeze(1)


        ################################################################################################################
        # Compute the pose
        ################################################################################################################

        # We can choose to use just the keypoint loss and not compute pose for the first few epochs.
        if epoch >= self.config['start_svd']:

            #  Check that we have enough inliers for all example sin the bach to compute pose.
            valid = kpt_valid[src_ind] & valid_pseudo & valid_inliers

            num_inliers = torch.sum(valid.squeeze(1), dim=1)[0]
            if torch.any(num_inliers < 6):
                raise RuntimeError('Too few inliers to compute pose: {}'.format(num_inliers))

            weights[valid == 0] = 0.0
            T_trg_src = self.svd_block(kpt_3D_src, kpt_3D_pseudo, svd_weights)


        ################################################################################################################
        # Compute the losses
        ################################################################################################################

        # Keypoint loss in different versions depending on using 2D coordinates, 3D coordinates, or a subset of the
        # ground truth pose to transform the target points in the plane.
        if 'keypoint_3D' in loss_types:
            valid = kpt_valid_pseudo & kpt_valid_src & valid_inliers   # B x 1 x N
            valid = valid.expand(batch_size, 4, valid.size(2))         # B x 4 x N
            keypoint_loss = mse_loss_fn(kpt_3D_pseudo[valid], kpt_3D_pseudo_gt[valid])
            keypoint_loss *= loss_weights['keypoint_3D']
            losses['keypoint_3D'] = keypoint_loss
            losses['total'] += keypoint_loss

        if 'keypoint_2D' in loss_types:
            valid = kpt_valid_pseudo & kpt_valid_src & valid_inliers  # B x 1 x N
            valid = valid.expand(batch_size, 2, valid.size(2))        # B x 2 x N
            keypoint_loss = mse_loss_fn(kpt_2D_pseudo[valid], kpt_2D_pseudo_gt[valid])
            keypoint_loss *= loss_weights['keypoint_2D']
            losses['keypoint_2D'] = keypoint_loss
            losses['total'] += keypoint_loss

        if ('keypoint_plane' in loss_types):
            valid = valid_pseudo & kpt_valid_src & valid_inliers   # B x 1 x N
            valid = valid.expand(batch_size, 2, valid.size(2))     # B x 4 x N
            kpt_3D_pseudo_gt_vehicle = T_trg_src_inl_gt.bmm(se3_inv(T_s_v).bmm(kpt_3D_src))  # Bx4xN
            kpt_3D_pseudo_vehicle = se3_inv(T_s_v).bmm(kpt_3D_pseudo)
            keypoint_loss = mse_loss_fn(kpt_3D_pseudo_vehicle[:, 0:2, :][valid],
                                        kpt_3D_pseudo_gt_vehicle[:, 0:2, :][valid])
            keypoint_loss *= loss_weights['keypoint_plane']
            losses['keypoint_plane'] = keypoint_loss
            losses['total'] += keypoint_loss

        # Pose loss either using the full 6DOF pose or a subset (x, y, heading).
        if epoch >= self.config['start_svd']:
            if ('pose' in loss_types):
                T_trg_src_gt = poses_se3[0]
                rot_loss, trans_loss = self.pose_loss(T_trg_src_gt, T_trg_src, mse_loss_fn)
                rotation_loss = loss_weights['rotation'] * rot_loss
                translation_loss = loss_weights['translation'] * trans_loss
                pose_loss = rotation_loss + translation_loss
                losses['rotation'] = rotation_loss
                losses['translation'] = translation_loss

            elif ('pose_plane' in loss_types):
                w = torch.zeros(6)
                w[0] = loss_weights['translation_x']
                w[1] = loss_weights['translation_y']
                w[5] = loss_weights['rotation_heading']
                w = torch.diag(w).unsqueeze(0).expand(batch_size, 6, 6).cuda()

                T_trg_src_gt = poses_se3[0]
                log_pose_err = se3_log(T_trg_src.bmm(torch.inverse(T_trg_src_gt))).unsqueeze(2)
                pose_loss = (1.0 / batch_size) * torch.sum(log_pose_err.transpose(2, 1).bmm(w).bmm(log_pose_err))

            losses['pose'] = pose_loss
            losses['total'] += pose_loss

        ################################################################################################################
        # Return the estimated pose and the loss
        ################################################################################################################

        if epoch >= self.config['start_svd']:
            return losses, T_trg_src
        else:
            # Haven't computed the pose yet, just the keypoint loss.
            return losses, poses_se3[0]


    def pose_loss(self, T, T_pred, loss_fn):
        batch_size = T.size(0)

        R_pred = T_pred[:, 0:3, 0:3]
        R = T[:, 0:3, 0:3]
        identity = torch.eye(3).unsqueeze(0).expand(batch_size, 3, 3).cuda()

        rot_loss = loss_fn(R_pred.transpose(2, 1).bmm(R), identity)
        trans_loss = loss_fn(T_pred[:, 0:3, 3], T[:, 0:3, 3])

        return rot_loss, trans_loss

    def print_loss(self, losses, epoch, iter):
        print('\nepoch {}:'.format(epoch, iter))

        for key in losses:
            print('{}: {}'.format(key, losses[key].item()))

        print('', flush=True)







