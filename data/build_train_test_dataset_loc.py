import argparse
import json
import math
import os
import pickle
import random
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils import data

from data.build_datasets import build_random_loc_dataset, build_sequential_loc_dataset
from src.utils.utils import MELData, compute_mean_std

torch.backends.cudnn.benchmark = True

def main(config):

    # Directory to store outputs to stdout/stderr.
    results_path = f"{config['home_path']}/results/dataset/{config['dataset_name']}/"
    # Directory where the data is stored (images, transforms text files).
    data_path = f"{config['data_path']}/"
    # File to store the generated training dataset.
    dataset_path = f"{config['home_path']}/datasets/{config['dataset_name']}.pickle"

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Print outputs to files
    # orig_stdout = sys.stdout
    # out_fl = 'out_data.txt'
    # fl = open(results_path + out_fl, 'w')
    # sys.stdout = fl
    # orig_stderr = sys.stderr
    # fe = open(results_path + 'err.txt', 'w')
    # sys.stderr = fe

    # Set up device, using GPU 0
    device = torch.device('cuda:{}'.format(0) if torch.cuda.device_count() > 0 else 'cpu')
    torch.cuda.set_device(0)

    start = time.time()

    num_samples_train = {}
    for path_name in config['train_paths']:
        num_samples_train[path_name] = math.floor(config['sampling_ratios_train'][path_name] * config['num_train_samples'])  

    num_samples_valid = {}
    for path_name in config['test_paths']:
        num_samples_valid[path_name] = math.floor(config['sampling_ratios_valid'][path_name] * config['num_validation_samples'])

    # Training and validation data
    train_ids, train_labels_se3, train_labels_log = build_random_loc_dataset(data_path,
                                                                             config['train_paths'],
                                                                             config['train_runs'],
                                                                             num_samples_train,
                                                                             config['max_temporal_len'])

    valid_ids, valid_labels_se3, valid_labels_log = build_random_loc_dataset(data_path,
                                                                             config['test_paths'],
                                                                             config['validation_runs'],
                                                                             num_samples_valid,
                                                                             config['max_temporal_len'])

    # Class for storing the data before we write it to a pickle file.
    mel_data = MELData()

    mel_data.train_ids = train_ids
    mel_data.train_labels_se3 = train_labels_se3
    mel_data.train_labels_log = train_labels_log

    mel_data.valid_ids = valid_ids
    mel_data.valid_labels_se3 = valid_labels_se3
    mel_data.valid_labels_log = valid_labels_log

    with open(dataset_path, 'wb') as f:
        pickle.dump(mel_data, f, pickle.HIGHEST_PROTOCOL)

    print(f'\nTraining data size: {len(train_ids)}')
    print(f'Validation data size: {len(valid_ids)}\n')
    print('Saved training and validation dataset', flush=True)
    print(f'Generating training and validation datasets took {time.time() - start} s\n')

    # Compute mean and std_dev for the training set to use for input normalization if desirable.
    mean, std_dev = compute_mean_std(mel_data, data_path, config['dataset']['height'], config['dataset']['width'])
    mel_data.mean = mean
    mel_data.std_dev = std_dev

    # Generate sequential datasets that can be used for testing.
    mel_data.paths = config['test_paths']
    for path_name in config['test_paths']:

        test_runs = config['test_runs'][path_name]

        # We localize different runs of a path against each other and so store the resulting data in lists.
        mel_data.test_ids[path_name] = []
        mel_data.test_labels_se3[path_name] = []
        mel_data.test_labels_log[path_name] = []

        # Localize each of the runs to each other (but not both ways or to themselves).
        for i in range(len(test_runs) - 1):

            teach_run = test_runs[i]
            repeat_runs = test_runs[(i + 1):]
            temporal_len = config['temporal_len'][path_name][str(teach_run)]
            test_ids, test_labels_se3, test_labels_log = build_sequential_loc_dataset(data_path,
                                                                                      path_name,
                                                                                      teach_run,
                                                                                      repeat_runs,
                                                                                      temporal_len)

            mel_data.test_ids[path_name] = mel_data.test_ids[path_name] + [test_ids]
            mel_data.test_labels_se3[path_name] = mel_data.test_labels_se3[path_name] + [test_labels_se3]
            mel_data.test_labels_log[path_name] = mel_data.test_labels_log[path_name] + [test_labels_log]

    with open(dataset_path, 'wb') as f:
        pickle.dump(mel_data, f, pickle.HIGHEST_PROTOCOL)

    print('\nSaved test dataset', flush=True)
    print(f'Generating full training, validation, and test dataset took {time.time() - start} s')

    # Stop writing outputs to files.
    # sys.stdout = orig_stdout
    # fl.close()
    # sys.stderr = orig_stderr
    # fe.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None, type=str,
                      help='config file path (default: None)')

    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    main(config)
