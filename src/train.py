import argparse
import copy
import json
import os
import pickle
import re
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils import data

from src.dataset import Dataset
from src.losses.losses import compute_loss
from src.model.unet import UNet
from src.utils.lie_algebra import se3_log, se3_exp
from src.utils.path_tools import integrate_path_vo, integrate_path_loc, integrate_path_teach
from src.utils.statistics import Statistics
from src.utils.stereo_camera_model import StereoCameraModel
from src.utils.transform2 import Transform
from src.utils.utils import prepare_device
from visualization.plots import Plotting

torch.backends.cudnn.benchmark = True


def excute_epoch(pipeline, net, data_loader, stats, epoch, mode, optimizer=None, scheduler=None):
    """
        Train with data for one epoch.

        Args:
            pipeline (Pipeline): the training pipeline object that performs a forward pass of the training pipeline and
                                 computes pose and losses.
            net (torch.nn.Module): neural network module.
            data_loader (torch.utils.data.DataLoader): data loader for training data.
            stats (TDRO):
            epoch (int): current epoch.
            mode (string): 'training' or 'validation'
            optimizer (torch.optim.Optimizer or None): optimizer or None if validation.
            scheduler(TODO or None): scheduler or None if validation.
    """

    start_time = time.time()

    if mode == 'training':
        net.train()
    else:
        net.eval()

    stats.epoch_reset()

    epoch_losses = {}
    errors = None
    # We want to print out the avg./max target pose absolute values for reference together with the pose errors.
    targets_total = torch.zeros(6)
    targets_max = torch.zeros(6)
    batch_size = -1
    num_batches, num_examples = 0, 0

    with torch.set_grad_enabled(mode == 'training'):
        for images, disparity, ids, poses_se3, poses_log in data_loader:

            losses = {}
            batch_size = images.size(0)
            if mode == 'training':
                optimizer.zero_grad()

            try:
                # Compute the loss and the output poses.
                losses, outputs_se3 = pipeline.forward(net, images, disparities, poses_se3, poses_log, epoch)

                if mode == 'training':
                    losses['total'].backward()

            except Exception as e:
                print(e)
                print("Ids: {}".format(ids))
                continue

            if mode == 'training':
                torch.nn.utils.clip_grad_norm(self.model.parameters(), max_norm=2.0, norm_type=2)
                optimizer.step()

            num_batches += 1
            num_examples += batch_size

            # Get the error in each of the 6 pose DOF.
            diff = se3_log(outputs_se3.inverse().bmm(poses_se3[0])).detach().cpu().numpy()
            if num_batches == 1:
                errors = diff
            else:
                errors = np.concatenate((errors, diff), axis=0)

            # Save the losses during the epoch.
            for loss_type in losses.keys():
                if loss_type in epoch_losses:
                    epoch_losses[loss_type] += losses[loss_type].item()
                else:
                    epoch_losses[loss_type] = losses[loss_type].item()

            targets_total += torch.sum(torch.abs(pose_se3), dim=0).detach().cpu()
            targets_max = torch.max(targets_max, torch.max(torch.abs(pose_se3.detach().cpu()), dim=0)[0])

    # Compute and print the statistics for the finished epoch.
    print(f'Epoch: {epoch}')

    # Compute the average training losses.
    for loss_type in epoch_losses.keys():
        epoch_losses[loss_type] = epoch_losses[loss_type] / float(num_batches)
        print(f'Average training loss, {loss_type}: {epoch_losses[k]:.6f}')

    # Only store and print pose errors if we have started computing pose (and are not just computing keypoint loss).
    if epoch >= self.config['start_svd']:

        # RMSE for each pose DOF.
        errors[:, 3:6] = np.rad2deg(errors[:, 3:6])
        epoch_errors = np.sqrt(np.mean(errors ** 2, axis=0)).reshape(1, 6)

        stats.add_epoch_stats(epoch_losses, epoch_errors)

        targets_avg = (targets_total * (1.0 / num_examples)).detach().cpu().numpy()
        targets_avg[3:6] = np.rad2deg(targets_avg[3:6])
        targets_max = targets_max.detach().cpu().numpy()
        targets_max[3:6] = np.rad2deg(targets_max[3:6])

        print(f'Pose RMSE by DOF: {epoch_errors}')
        print(f'Average pose targets by DOF: {targets_avg}')
        print(f'Max pose targets by DOF: {targets_max}')

        print(f'Epoch duration: {time.time() - start} seconds. \n')

    # Run if we are training, and we are using a scheduler.
    if scheduler is not None:
        scheduler.step()

    return epoch_losses['total'], stats


def plot_epoch_statistics(self):
    """
    Plot the losses from training and validation
    """
    train_epoch_stats = self.train_stats.get_epoch_stats()
    valid_epoch_stats = self.valid_stats.get_epoch_stats()
    # valid_epoch_stats = self.train_stats.get_epoch_stats()

    self.plotting.plot_epoch_losses(train_epoch_stats[0], valid_epoch_stats[0])
    self.plotting.plot_epoch_errors(train_epoch_stats[1], valid_epoch_stats[1])


def train(pipeline, net, optimizer, scheduler, train_loader, validation_loader, start_epoch, min_validation_loss,
          train_stats, validation_stats, config, plotting, results_path, checkpoint_path):
    """
        Run the training loop.

        Args:
            pipeline (Pipeline): the training pipeline object that performs a forward pass of the training pipeline and
                                 computes pose and losses.
            net (torch.nn.Module): neural network module.
            optimizer (torch.optim.Optimizer): training optimizer.
            train_loader (torch.utils.data.DataLoader): data loader for training data.
            validation_loader (torch.utils.data.DataLoader): data loader for validation data.
            start_epoch (int): epoch we start training from, 0 if starting from scratch.
            min_validation_loss (float): the minimum validation loss during training.
            train_stats (TODO):
            validation_stats (TODO):
            config (dict): configurations for model training.
            plotting (Plotting): object used to plot training statistics.
            results_path (string): directory for storing results (outputs, plots).
            checkpoint_path (string)L directory to store the checkpoint.
    """

    # Early stopping keeps track of when we stop training (validation loss does not decrease) and when we store a
    # checkpoint (each time validation loss decreases below or is equal to the minimum value).
    last_save_epoch = start_epoch - 1 if start_epoch > 0 else 0
    early_stopping = EarlyStopping(config['trainer']['patience'], min_validation_loss, last_save_epoch)

    # The training loop.
    for epoch in range(start_epoch, config['trainer']['max_epochs']):

        train_loss = self.execute_epoch(pipeline, net, optimizer, train_loader, train_stats, epoch, mode='training')
        validation_loss = self.execute_epoch(pipeline, net, optimizer, train_loader, train_stats, epoch, mode='validation')

        # Only start checking the loss after we start computing the pose loss. Sometimes we only compute keypoint loss
        # for the first few epochs.
        if epoch >= config['start_svd']:

            # Plot the losses during training and validations.
            self._plot_epoch_statistics()

            stop, min_validation_loss = early_stopping.check_stop(validation_loss,
                                                                  net,
                                                                  optimizer,
                                                                  checkpoint_path,
                                                                  train_stats,
                                                                  validation_stats,
                                                                  epoch)

            # Stop the training loop if we have exceeded the patience.
            if stop:
                break


def main(config):
    """
        Set up everything needed for training and call the function to run the training loop.

        Args:
            config (dict): configurations for training the network.
    """

    results_path = '{0}/results/{}'.format(config['home_path'], config['experiment_name'])
    networks_path = '{0}/networks'.format(config['home_path'])
    data_path = '{0}/data'.format(config['data_path'])
    datasets_path = '{0}/datasets'.format(config['home_path'])

    checkpoint_name = config['checkpoint']
    network_name = config['network_name']
    dataset_name = config['dataset_name']

    dataset_path = '{0}/{1}.pickle'.format(datasets_path, dataset_name)
    checkpoint_path = '{0}/{1}.pth'.format(networks_path, checkpoint_name)

    # Print outputs to files
    orig_stdout = sys.stdout
    out_fl = 'out_train.txt'
    fl = open(results_path + out_fl, 'w')
    sys.stdout = fl
    orig_stderr = sys.stderr
    fe = open(results_path + 'err_train.txt', 'w')
    sys.stderr = fe

    # Record the config settings
    print(config)

    # Load the data
    dataloader_params = config['data_loader']
    dataset_params = config['dataset']
    dataset_params['data_dir'] = data_path

    loc_data = None
    with open(dataset_path, 'rb') as handle:
        loc_data = pickle.load(handle)

    # Training data generator
    train_set = Dataset(config['num_frames'], False, **dataset_params)
    train_sampler = RandomSampler(train_set, replacement=True, num_samples=10000)
    train_set.load_mel_data(loc_data, 'train')
    train_loader = data.DataLoader(train_set, sampler=train_sampler, **dataloader_params)

    # Validation data generator
    validation_set = Dataset(config['num_frames'], False, **dataset_params)
    validation_sampler = RandomSampler(validation_set, replacement=True, num_samples=2500)
    validation_set.load_mel_data(loc_data, 'validate')
    validation_loader = data.DataLoader(validation_set, sampler=validation_sampler, **dataloader_params)

    # Set up device, using GPU 0
    device = torch.device('cuda:{}'.format(0) if torch.cuda.device_count() > 0 else 'cpu')
    torch.cuda.set_device(0)

    # Set training pipeline
    training_pipeline = Pipeline(config)
    training_pipeline = training_pipeline.to(device)

    # Set up the network, optimizer, and scheduler
    net = UNet(config['network']['num_channels'],
               config['network']['num_classes'],
               config['network']['layer_size'])

    optimizer = torch.optim.Adam(net.parameters(), lr=config['optimizer']['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=config['scheduler']['step_size'],
                                                gamma=config['scheduler']['gamma'])

    # Variables for keeping track of training progress.
    start_epoch = 0
    min_validation_loss = 10e9
    train_stats = Statistics()
    validation_stats = Statistics()

    # Resume training from checkpoint if available
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

        start_epoch = checkpoint['epoch'] + 1
        min_validation_loss = checkpoint['val_loss'] if 'val_loss' in checkpoint.keys() else 10e9
        train_stats = checkpoint['train_stats'] if 'train_stats' in checkpoint.keys() else Statistics()
        validation_stats = checkpoint['valid_stats'] if 'valid_stats' in checkpoint.keys() else Statistics()

    net.cuda()

    plotting = Plotting(results_path)

    train(training_pipeline, net, train_loader, validation_loader, start_epoch, min_validation_loss,
          train_stats, validation_stats, config, plotting, results_path, checkpoint_path)

    # Stop writing outputs to files.
    sys.stdout = orig_stdout
    fl.close()
    sys.stderr = orig_stderr
    fe.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None, type=str,
                        help='config file path (default: None)')

    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    main(config)
