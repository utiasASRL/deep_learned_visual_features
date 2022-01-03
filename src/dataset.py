import os
import random
import re

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms

from src.utils.lie_algebra import se3_log, se3_inv
from src.utils.transform import Transform

class Dataset(data.Dataset):
    """
        Dataset for the localization data.
    """

    def __init__(self, data_dir, height=384, width=512, crop_height=384, crop_width=512,
                 use_normalization=True, use_crop=True, use_disparity=True):

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

        # Create transforms to apply to all images
        self.image_transforms = transforms.Compose([
            transforms.ToTensor()
        ])

        self.normalize_transforms = None

    def load_mel_data(self, mel_data, mode, path_name='', index=None):

        self.normalize_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mel_data.mean, mel_data.std_dev)
        ])
        
        if mode == 'train':
            self.list_ids = mel_data.train_ids
            self.labels_se3 = mel_data.train_labels_se3
            self.labels_log = mel_data.train_labels_log

        elif mode == 'validate':
            self.list_ids = mel_data.valid_ids
            self.labels_se3 = mel_data.valid_labels_se3
            self.labels_log = mel_data.valid_labels_log

        elif mode == 'test':
            self.list_ids = mel_data.test_ids[path_name][index]
            self.labels_se3 = mel_data.test_labels_se3[path_name][index]
            self.labels_log = mel_data.test_labels_log[path_name][index]

        else:
            raise ValueError('Dataset, load_mel_data: mode must be set to train, validate, or test.')

    def __len__(self):
        '''Denotes the total number of samples'''
        return len(self.list_ids)

    def __getitem__(self, index):
        '''Generates one sample of data
           Data sample consist of a set of consecutive images together with a target VO velocity.
           The length of the sequence can be adjusted.
           We can choose whether to use monocular or stereo images.
        '''

        # Select sample
        data_id = self.list_ids[index]

        # Get the path, run, and vertex for the sample.
        data_info = re.findall('\w+', data_id)
        path_name = data_info[0]
        run_ids = data_info[1::2]
        vertex_ids = data_info[2::2]

        self.im_height = self.crop_height if self.use_crop else self.height
        self.im_width = self.crop_width if self.use_crop else self.width

        # Indices to use for cropping the images
        i, j = 0, 0

        X = torch.zeros(num_channels, self.im_height, self.im_width)   # Images
        D = torch.zeros(num_frames_use, self.im_height, self.im_width) # Disparity

        self.add_images(X, 0, path_name, run_ids[0], vertex_ids[0], i, j, self.use_normalization) # Source image
        self.add_images(X, 1, path_name, run_ids[1], vertex_ids[1], i, j, self.use_normalization) # Target image

        if self.use_disparity:
            self.add_disparity(D, 0, path_name, run_ids[0], vertex_ids[0]) # Source disparity
            self.add_disparity(D, 1, path_name, run_ids[1], vertex_ids[1]) # Target disparity

        # Pose transforms
        y_se3 = self.labels_se3[data_id]
        y_log = self.labels_log[data_id]

        return X, D, data_id, y_se3, y_log

    def add_images(self, X, ind, path_name, run_id, vertex_id, i, j, normalize_img):
        ''' Add left, right and reference images to the dataset sample according to the
            specified dataset settings. '''

        # Stereo pair of RGB images (2 x 3 channels).
        start = ind * 6
        self.add_image(X, start, 'left', path_name, run_id, vertex_id, i, j, normalize_img)
        self.add_image(X, start + 3, 'right', path_name, run_id, vertex_id, i, j, normalize_img)

    def add_image(self, X, start_ind, loc, path_name, run_id, vertex_id, i, j, normalize_img):
        ''' Add one image to the dataset sample.'''

        img_file = f"{self.data_dir}path_{path_name}_processed/run_{run_id.zfill(6)}/" \
                   f"images/{loc}/{vertex_id.zfill(6)}.png"

        img = Image.open(img_file)

        if self.use_crop:
            img = transforms.functional.crop(img, 0.0, self.width - self.im_width, self.im_height, self.im_width)

        # Turn the image into a tensor and normalize if required.
        if normalize_img:
            X[start_ind:start_ind + 3, :, :] = self.normalize_transforms(img)
        else:
            X[start_ind:start_ind + 3, :, :] = self.image_transforms(img)

    def get_disparity(self, left_img, right_img, pyr_level):

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
        
        if self.resize_images:
            # Uses bilinear interpolation when downsampling.
            disp = cv2.resize(disp,
                              (self.desired_width, self.desired_height),
                              fx=0,
                              fy=0,
                              interpolation = cv2.INTER_NEAREST)

        # Adjust values close to or equal to zero, which would cause problems for depth calculation.
        disp[(disp < 1e-4) & (disp >= 0.0)] = 1e-4

        return torch.from_numpy(disp)

    def add_disparity(self, D, index, path_name, run_id, vertex_id):

        img_file_left = f"{self.data_dir}path_{path_name}_processed/run_{run_id.zfill(6)}/images/" \
                        f"left/{vertex_id.zfill(6)}.png"
        img_file_right = f"{self.data_dir}path_{path_name}_processed/run_{run_id.zfill(6)}/images/" \
                        f"right/{vertex_id.zfill(6)}.png"

        left_img = np.uint8(cv2.imread(img_file_left, 0))
        right_img = np.uint8(cv2.imread(img_file_right, 0))
            
        # Compute disparity using OpenCV.
        disparity = self.get_disparity(left_img, right_img, 0)

        if self.use_crop:
            disparity = disparity[:self.im_height, self.desired_width - self.im_width:]

        D[index, :, :] = disparity