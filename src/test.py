import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils import data

from src.dataset import Dataset
from src.utils.lie_algebra import se3_log, se3_exp
from src.utils.statistics import Statistics
from visualization.plots import Plotting

torch.backends.cudnn.benchmark = True

def rmse(out_dict, trg_dict):

    for run in out_dict.keys():

        out_mat = torch.from_numpy(np.stack(out_dict[run], axis=0))
        trg_mat = torch.from_numpy(np.stack(trg_dict[run], axis=0))

        diff_mat = trg_mat.bmm(out_mat.inverse())
        R = diff_mat[:, 0:3, 0:3]
        diff_r = diff_mat[:, 0:3, 3].numpy()

        err_tr = np.sqrt((diff_r[:, 0] * diff_r[:, 0]) + (diff_r[:, 1] * diff_r[:, 1]) + (diff_r[:, 2] * diff_r[:, 2]))
        rmse_tr = np.sqrt(np.mean(err_tr**2, axis=0))

        d = torch.acos((0.5 * (R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2] - 1.0)).clamp(-1 + 1e-6, 1 - 1e-6)).numpy()
        rmse_rot = np.sqrt(np.mean(np.rad2deg(d)**2, axis=0))

        print("RMSE, run {}, tr: {}".format(run, rmse_tr))
        print("RMSE, run {}, rot: {}\n".format(run, rmse_rot))

def test_model(pipeline, net, data_loader, stats):
    """
        Run the test.

        Args:
            pipeline (Pipeline): the training pipeline object that performs a forward pass of the training pipeline and
                                 computes pose and losses.
            net (torch.nn.Module): neural network module.
            data_loader (torch.utils.data.DataLoader): data loader for training data.
            stats (Statistics): object to keep track of test losses and errors.
    """
    start_time = time.time()

    net.eval()
    stats.epoch_reset()

    test_losses = {}
    errors = None
    # We want to print out the avg./max target pose absolute values for reference together with the pose errors.
    targets_total = torch.zeros(6)
    targets_max = torch.zeros(6)
    batch_size = -1
    num_batches, num_examples = 0, 0

    with torch.no_grad():
        for images, disparity, ids, poses_se3, poses_log in data_loader:

            losses = {}
            batch_size = images.size(0)

            try:
                # Compute the loss and the output poses.
                losses, outputs_se3 = pipeline.forward(net, images, disparities, poses_se3, poses_log, epoch)

            except Exception as e:
                print(e)
                print("Ids: {}".format(ids))
                continue

            num_batches += 1
            num_examples += batch_size

            # Get the error in each of the 6 pose DOF.
            diff = se3_log(outputs_se3.inverse().bmm(poses_se3[0])).detach().cpu().numpy()
            if num_batches == 1:
                errors = diff
            else:
                errors = np.concatenate((errors, diff), axis=0)

            # Save the losses.
            for loss_type in losses.keys():
                if loss_type in test_losses:
                    test_losses[loss_type] += losses[loss_type].item()
                else:
                    test_losses[loss_type] = losses[loss_type].item()

            targets_total += torch.sum(torch.abs(pose_se3), dim=0).detach().cpu()
            targets_max = torch.max(targets_max, torch.max(torch.abs(pose_se3.detach().cpu()), dim=0)[0])

            # Collect the poses so we can plot the results later.
            outputs_se3_np = outputs_se3.detach().cpu().numpy()  # Bx4x4
            targets_se3_np = targets_se3.detach().cpu().numpy()
            # kpt_inliers = torch.sum(saved_data['valid_inliers'][(0, 1)], dim=2).detach().cpu().numpy()  # B x 1

            # Loop over each datas sample in the batch.
            for k in range(outputs_se3_np.shape[0]):
                sample_id = ids[k]
                _, live_run_id, _, map_run_id, _ = re.findall('\w+', sample_id)

                stats.add_sample_id(live_run_id, sample_id)
                stats.add_live_run_id(live_run_id)
                stats.add_map_run_id(map_run_id, live_run_id)

                stats.add_outputs_targets_transforms(live_run_id, outputs_se3_np[k, :, :], targets_se3_np[k, :, :])
                # stats.add_kpt_inliers(live_run_id, kpt_inliers[k, 0])

    # Compute the average test losses.
    for loss_type in test_losses.keys():
        test_losses[loss_type] = test_losses[loss_type] / float(num_batches)
        print(f'Average test loss, {loss_type}: {epoch_losses[k]:.6f}')

    # RMSE for each pose DOF.
    errors[:, 3:6] = np.rad2deg(errors[:, 3:6])
    test_errors = np.sqrt(np.mean(errors ** 2, axis=0)).reshape(1, 6)

    stats.add_epoch_stats(test_losses, test_errors)

    targets_avg = (targets_total * (1.0 / num_examples)).detach().cpu().numpy()
    targets_avg[3:6] = np.rad2deg(targets_avg[3:6])
    targets_max = targets_max.detach().cpu().numpy()
    targets_max[3:6] = np.rad2deg(targets_max[3:6])

    print(f'Pose RMSE by DOF: {epoch_errors}')
    print(f'Average pose targets by DOF: {targets_avg}')
    print(f'Max pose targets by DOF: {targets_max}')

    print(f'Epoch duration: {time.time() - start} seconds. \n')

    return stats

def test(pipeline, net, test_loaders, results_path):
    """
        Test the model by localizing different runs to each other from one or more paths. Print the pose errors and
        losses and plot the target and estimated poses for each DOF for each test.

        Args:
            pipeline (Pipeline): the training pipeline object that performs a forward pass of the training pipeline and
                                 computes pose and losses.
            net (torch.nn.Module): neural network module.
            test_loaders (dict): maps path names to data loaders for test data.
            results_path (string): directory for storing results (outputs, plots).
    """
    # Helper class for plotting results.
    plotting = Plotting(results_path)

    # The data loaders are stored in a dict that maps from path name to a list of data loaders as we may test using
    # data from more than one path. Also, for each path we localize different runs against each other. One run is used
    # as the map and then one or more other runs are localized to this map run. One individual data loader contains all
    # the samples for all live localized back to one map run.
    for path_name in test_loaders.keys():

        print(f'Testing path: {path_name}')

        start = time.time()
        path_stats = Statistics()

        # Loop over each data loader (one data loader per map run we localize to).
        for i in range(len(test_loaders[path_name])):

            path_stats = test_model(pipeline, net, test_loaders[path_name][i], path_stats)

            outputs = path_stats.get_outputs()
            targets = path_stats.get_targets()
            map_run_id = path_stats.get_map_run_id()
            live_run_ids = path_stats.get_live_run_ids()
            sample_ids = path_stats.get_sample_ids()
            kpt_inliers = path_stats.get_kpt_inliers()

            # Plot each DOF of the estimated and target poses for each pair of live run localized to map run.
            # plotting.plot_outputs(outputs, outputs_tr, targets, targets_tr, path_name, 0, map_run_id, data_type,
            #                       ylim=[-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])

        print(f'Test time for path {path_name}: {time.time() - start}')

def main(config):
    """
        Set up everything needed for training and call the function to run the training loop.

        Args:
            config (dict): configurations for training the network.
    """
    results_path = f"{config['home_path']}/results/{config['experiment_name']}"
    networks_path = f"{config['home_path']}/networks"
    data_path = f"{config['data_path']}/data"
    datasets_path = f"{config['home_path']}/datasets"

    checkpoint_name = config['checkpoint']
    network_name = config['network_name']
    dataset_name = config['dataset_name']

    dataset_path = f'{datasets_path}/{dataset_name}.pickle'
    checkpoint_path = f'{networks_path}/{checkpoint_name}.pth'

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Print outputs to files
    orig_stdout = sys.stdout
    out_fl = 'out_test.txt'
    fl = open(results_path + out_fl, 'w')
    sys.stdout = fl
    orig_stderr = sys.stderr
    fe = open(results_path + 'err_test.txt', 'w')
    sys.stderr = fe

    # Record the config settings.
    print(config)

    # Load the data.
    dataloader_params = config['data_loader']
    dataloader_params['shuffle'] = False
    dataset_params = config['dataset']
    dataset_params['data_dir'] = data_path

    localization_data = None
    with open(dataset_path, 'rb') as handle:
        localization_data = pickle.load(handle)

    # The localization testing may contain data from different paths.
    path_loaders = {}
    for path_name in localization_data.paths:

        # We localize a set of runs against each other. One run is used as thep map and several other runs are used as
        # 'live' runs and localized to the map run. We create one data loader for each map run. Hence, the loader will
        # hold all data samples of localizing each live run to the map run.
        num_map_runs = len(localization_data.test_ids[path_name]) # The number of data loaders we will need.
        path_loaders[path_name] = []

        index = 0
        while index < num_map_runs:

            # Check that the dataset is not empty before adding.
            if len(localization_data.test_ids[path_name][index]) > 0:

                test_set = Dataset(**dataset_params)
                test_set.load_mel_data(localization_data, 'testing', path_name, index)
                test_loader = data.DataLoader(test_set, **dataloader_params)
                path_loaders[path_name] += [test_loader]

            index += 1

    # checkpoint = torch.load(checkpoint_path)
    # model.load_state_dict(checkpoint['model_state_dict'])
    #
    # # Store state dict for the network model only.
    # new_state_dict = {}
    # for name, param in model.state_dict().items():
    #     if 'net' in name:
    #         new_state_dict[name.replace('net.', '')] = param
    # torch.save({'model_state_dict': new_state_dict}, '/home/mgr/results/network_snow_391_16_h16w16.pth')

    # Set up device, using GPU 0.
    device = torch.device('cuda:{}'.format(0) if torch.cuda.device_count() > 0 else 'cpu')
    torch.cuda.set_device(0)

    # Set training pipeline.
    testing_pipeline = Pipeline(config)
    testing_pipeline = testing_pipeline.cuda()

    # Set up the network.
    net = UNet(config['network']['num_channels'],
               config['network']['num_classes'],
               config['network']['layer_size'])

    # Load the network weights from a checkpoint.
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise RuntimeError(f'The specified checkpoint path does not exists: {checkpoint_path}')

    net.cuda()

    test(testing_pipeline, net, path_loaders, config, results_path)

    # Stop writing outputs to file.
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