import time

import numpy as np
import torch
from torch.utils import data

from src.dataset import Dataset

class MELData():
    """
        Class for storing the training, validation, and test data that we generate from the VT&R Multi-Experience
        Localization (MEL) data.
    """

    def __init__(self):
        self.mean = [0.0, 0.0, 0.0]
        self.std_dev = [0.0, 0.0, 0.0]

        self.train_labels_se3 = {}
        self.train_labels_log = {}
        self.train_ids = []

        self.valid_labels_se3 = {}
        self.valid_labels_log = {}
        self.valid_ids = []

        self.paths = []
        self.test_labels_se3 = {}
        self.test_labels_log = {}
        self.test_ids = {}

def compute_mean_std(mel_data, data_dir, height, width):
    """
        Compute the mean and standard deviation for each channel of the images in a dataset.

        Args:
            mel_data (MELData): the object storing the images.
            data_dir (string): the directory where the images are stored
            height (int): image height.
            width (int): image width.

        Returns:
            pop_mean_final (list[float]): the mean for each of the three RGB channels for the images.
            pop_std_final (list[float]): the standard deviation for each of the three RGB channels for the images.
    """

    start = time.time()

    image_params = {'data_dir': data_dir,
                    'height': height,
                    'width': width,
                    'crop_height': height,
                    'crop_width': width,
                    'use_normalization': False,
                    'use_crop': False,
                    'use_disparity': False,
                    }

    num_channels = 12
    batch_size = 128
    
    params = {'batch_size': batch_size,
              'shuffle': False,
              'num_workers': 12}

    training_set = Dataset(**image_params)
    training_set.load_mel_data(mel_data, 'training')
    training_generator = data.DataLoader(training_set, **params)

    fst_moment = torch.zeros(num_channels)
    snd_moment = torch.zeros(num_channels)

    fst_moment = fst_moment.cuda()
    snd_moment = snd_moment.cuda()

    i = 0
    cnt = 0
    for images, _, _, _, _ in training_generator:
        i += 1

        images = images[:, 0:num_channels, :, :].cuda()

        b, c, h, w = images.size()
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0,2,3])
        sum_of_square = torch.sum(images ** 2, dim=[0,2,3])
        fst_moment = ((cnt * fst_moment) + sum_) / (cnt + nb_pixels)
        snd_moment = ((cnt * snd_moment) + sum_of_square) / (cnt + nb_pixels)

        cnt += nb_pixels

    pop_mean = fst_moment
    pop_std = torch.sqrt(snd_moment - (fst_moment ** 2))

    pop_mean_final = [0, 0, 0]
    pop_std_final = [0, 0, 0]

    num_comp = int(num_channels / 3.0)

    for i in range(3):
        for j in range(num_comp):
            pop_mean_final[i] += pop_mean[(j * 3) + i].item()
            pop_std_final[i] += pop_std[(j * 3) + i].item()

        pop_mean_final[i] = pop_mean_final[i] / num_comp
        pop_std_final[i] = pop_std_final[i] / num_comp

    print(f'Training images mean: {pop_mean_final}')
    print(f'Training images std: {pop_std_final}')
    print(f'Computing mean/std took: {time.time() - start}')

    return pop_mean_final, pop_std_final
