import os
import random
import re

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms

class Dataset(data.Dataset):
    """
        Dataset for the VT&R localization data.
    """

    def __init__(self, data_dir, height=384, width=512, crop_height=384, crop_width=512,
                 use_normalization=False, use_crop=False, use_disparity=True):
        """
            Initialize the Dataset class with the necessary parameters.

            Args:
                data_dir (string): the directory where the images are stored.
                height (int): height of the images.
                width (int): width of the images.
                crop_height (int): desired height of the images if they should be cropped.
                crop_width (int): desired width of the images if they should be cropped.
                use_normalization (bool): whether to normalize the images.
                use_crop (bool): whether to crop the images.
                use_disparity (bool): whether to generate disparity for stereo pairs.
        """

        self.list_ids = []
        self.labels_se3 = {}
        self.labels_log = {}
        self.data_dir = data_dir
        self.height = height
        self.width = width
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.use_normalization = use_normalization
        self.use_crop = use_crop
        self.use_disparity = use_disparity

        # Create transforms to apply to images
        self.image_transforms = transforms.Compose([
            transforms.ToTensor()
        ])

        self.normalize_transforms = None

    def load_mel_data(self, mel_data, mode, path_name='', index=None):
        """
            Load the dataset that we have created ahead of time.

            Args:
                mel_data (MELData): object that stores the data sample ids and labels.
                mode (string): a string with value 'training', 'validation', or 'testing'.
                path_name (string, optional): if the data is used for testing, the name of the path we want to test.
                index (int, optional): if the data is used for testing, we have several test paths, the index indicates
                                       which one to use.
        """

        # Set the mean and std_dev if we want to normalize the data.
        self.normalize_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mel_data.mean, mel_data.std_dev)
        ])
        
        if mode == 'training':
            self.list_ids = mel_data.train_ids
            self.labels_se3 = mel_data.train_labels_se3
            self.labels_log = mel_data.train_labels_log

        elif mode == 'validation':
            self.list_ids = mel_data.valid_ids
            self.labels_se3 = mel_data.valid_labels_se3
            self.labels_log = mel_data.valid_labels_log

        elif mode == 'testing':
            self.list_ids = mel_data.test_ids[path_name][index]
            self.labels_se3 = mel_data.test_labels_se3[path_name][index]
            self.labels_log = mel_data.test_labels_log[path_name][index]

        else:
            raise ValueError('Dataset, load_mel_data: mode must be set to train, validate, or test.')

    def __len__(self):
        """
            The number of samples in the dataset.
        """
        return len(self.list_ids)

    def __getitem__(self, index):
        """
            Load a sample from the dataset

            Args:
                index (int): index of the sample to load.

            Returns:
                X (torch.tensor): two RGB stereo pairs of images stacked together, one source image pair and one target
                                  image pair (Bx12xHxW).
                D (torch.tensor): two disparity images, computed from the two pairs of stereo images (Bx2xHW).
                data_id (string): the id for the daa sample on the form pathname_runidsrc_poseidsrc_runidtrg_poseidtrg.
                y_se3 (torch.tensor): the relative pose from the source to target, T_trg_src, given as a 4x4 matrix
                                      (Bx4x4).
                y_log (torch.tensor): the relative pose from the source to target given as a length 6 vector (Bx6).
        """

        # Select sample
        data_id = self.list_ids[index]

        # Get the path name, run id, and pose id for the sample. Combining the run and pose id gives us the id of
        # a given vertex in the pose graph that the data was sampled from.
        data_info = re.findall('\w+', data_id)
        path_name = data_info[0]
        run_ids = data_info[1::2]
        pose_ids = data_info[2::2]

        self.im_height = self.crop_height if self.use_crop else self.height
        self.im_width = self.crop_width if self.use_crop else self.width

        # Indices to use for cropping the images
        i, j = 0, 0

        X = torch.zeros(num_channels, self.im_height, self.im_width)   # Images
        D = torch.zeros(num_frames_use, self.im_height, self.im_width) # Disparity

        self.add_images(X, 0, path_name, run_ids[0], pose_ids[0], i, j, self.use_normalization) # Source image
        self.add_images(X, 1, path_name, run_ids[1], pose_ids[1], i, j, self.use_normalization) # Target image

        if self.use_disparity:
            self.add_disparity(D, 0, path_name, run_ids[0], pose_ids[0]) # Source disparity
            self.add_disparity(D, 1, path_name, run_ids[1], pose_ids[1]) # Target disparity

        # Pose transforms
        y_se3 = self.labels_se3[data_id]
        y_log = self.labels_log[data_id]

        return X, D, data_id, y_se3, y_log

    def add_images(self, X, ind, path_name, run_id, pose_id, i, j, normalize_img):
        """
            Add a stereo pair of images to the images tensor.

            Args:
                X (torch.tensor): the tensor to hold the RGB stereo pairs of images (Bx12xHxW).
                ind (int): index indicating which image to add.
                path_name (string): name of the path the images are taken from.
                run_id (int): id of the run the images are taken from.
                pose_id (int): id of the pose along the given run that the images are taken from.
                i (int): index used for image cropping.
                j (int): index used for image cropping.
                normalize_img (bool): whether to normalize the image.
        """
        # Stereo pair of RGB images (2 x 3 channels).
        start = ind * 6
        self.add_image(X, start, 'left', path_name, run_id, pose_id, i, j, normalize_img)
        self.add_image(X, start + 3, 'right', path_name, run_id, pose_id, i, j, normalize_img)

    def add_image(self, X, start_ind, loc, path_name, run_id, pose_id, i, j, normalize_img):
        """
            Add one image to the images tensor.

            Args:
                X (torch.tensor): the tensor to hold the RGB stereo pairs of images (Bx12xHxW).
                start_ind (int): the index of the tensor where we start adding the images channels.
                loc (string): 'left' or 'right' insicating which image in the stereo pair we are adding.
                path_name (string): name of the path the images are taken from.
                run_id (int): id of the run the images are taken from.
                pose_id (int): id of the pose along the given run that the images are taken from.
                i (int): index used for image cropping.
                j (int): index used for image cropping.
                normalize_img (bool): whether to normalize the image.
        """
        # Generate the image file name based on the id of the vertex the image belongs to.
        img_file = f"{self.data_dir}path_{path_name}_processed/run_{run_id.zfill(6)}/" \
                   f"images/{loc}/{pose_id.zfill(6)}.png"

        img = Image.open(img_file)

        if self.use_crop:
            img = transforms.functional.crop(img, 0.0, self.width - self.im_width, self.im_height, self.im_width)

        # Turn the image into a tensor and normalize if required.
        if normalize_img:
            X[start_ind:start_ind + 3, :, :] = self.normalize_transforms(img)
        else:
            X[start_ind:start_ind + 3, :, :] = self.image_transforms(img)

    def get_disparity(self, left_img, right_img):
        """
            Create the disparity image using functions from OpenCV.

            Args:
                left_img (numpy.uint8): left stereo image.
                right_img (numpy.uint8): right stereo image.
        """
        stereo = cv2.StereoSGBM_create(minDisparity = 0,
                                       numDisparities = 48, 
                                       blockSize = 5, 
                                       preFilterCap = 30, 
                                       uniquenessRatio = 20, 
                                       P1 = 200, 
                                       P2 = 800, 
                                       speckleWindowSize = 200, 
                                       speckleRange = 1, 
                                       disp12MaxDiff = -1)

        disp = stereo.compute(left_img, right_img)
        disp  = disp.astype(np.float32) / 16.0

        # Adjust values close to or equal to zero, which would cause problems for depth calculation.
        disp[(disp < 1e-4) & (disp >= 0.0)] = 1e-4

        return torch.from_numpy(disp)

    def add_disparity(self, D, index, path_name, run_id, pose_id):
        """
            Add a disparity image to the tensor holding the disparity images.

            Args:
                D (torch.tensor): the tensor holding the disparity images.
                index (int): the index of the channel where we will add the disparity image in D.
                path_name (string): name of the path the images are taken from.
                run_id (int): id of the run the images are taken from.
                pose_id (int): id of the pose along the given run that the images are taken from.
        """
        # Generate the image file names based on the id of the vertex the image belongs to.
        img_file_left = f"{self.data_dir}path_{path_name}_processed/run_{run_id.zfill(6)}/images/" \
                        f"left/{pose_id.zfill(6)}.png"
        img_file_right = f"{self.data_dir}path_{path_name}_processed/run_{run_id.zfill(6)}/images/" \
                        f"right/{pose_id.zfill(6)}.png"

        left_img = np.uint8(cv2.imread(img_file_left, 0))
        right_img = np.uint8(cv2.imread(img_file_right, 0))
            
        # Compute disparity using OpenCV.
        disparity = self.get_disparity(left_img, right_img, 0)

        if self.use_crop:
            disparity = disparity[:self.im_height, self.desired_width - self.im_width:]

        D[index, :, :] = disparity