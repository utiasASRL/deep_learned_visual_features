import argparse
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
from torch.utils.data.sampler import RandomSampler

from src.dataset import Dataset
from src.model.pipeline import Pipeline
from src.model.unet import UNet
from src.utils.lie_algebra import se3_log
from src.utils.statistics import Statistics
from visualization.plots import Plotting

torch.backends.cudnn.benchmark = True

def rmse(outputs_se3, targets_se3):
    """
        Compute the rotation and translation RMSE for the SE(3) poses provided. Compute RMSE for ich live run
        individually.

        Args:
            outputs_se3 (dict): map from id of the localized live run to a list of estimated pose transforms
                                represented as 4x4 numpy arrays.
            outputs_se3 (dict): map from id of the localized live run to a list of ground truth pose transforms
                                represented as 4x4 numpy arrays.
    """

    for live_run_id in outputs_se3.keys():

        out_mat = torch.from_numpy(np.stack(outputs_se3[live_run_id], axis=0))
        trg_mat = torch.from_numpy(np.stack(targets_se3[live_run_id], axis=0))

        # Get the difference in pose by T_diff = T_trg * inv(T_src)
        diff_mat = trg_mat.bmm(out_mat.inverse())
        diff_R = diff_mat[:, 0:3, 0:3]
        diff_tr = diff_mat[:, 0:3, 3].numpy()

        err_tr_sq = (diff_tr[:, 0] * diff_tr[:, 0]) + (diff_tr[:, 1] * diff_tr[:, 1]) + (diff_tr[:, 2] * diff_tr[:, 2])
        rmse_tr = np.sqrt(np.mean(err_tr_sq, axis=0))

        d = torch.acos((0.5 * (diff_R[:, 0, 0] + diff_R[:, 1, 1] + diff_R[:, 2, 2] - 1.0)).clamp(-1 + 1e-6, 1 - 1e-6))
        rmse_rot = np.sqrt(np.mean(np.rad2deg(d.numpy())**2, axis=0))

        print(f'RMSE, live_run_id {live_run_id}, translation: {rmse_tr}')
        print(f'RMSE, live_run_id {live_run_id}, rotation: {rmse_rot}\n')

def test_model(pipeline, net, data_loader, dof):
    """
        Run the test.

        Args:
            pipeline (Pipeline): the training pipeline object that performs a forward pass of the training pipeline and
                                 computes pose and losses.
            net (torch.nn.Module): neural network module.
            data_loader (torch.utils.data.DataLoader): data loader for training data.
            dof (list[int]): indices of the pose DOF (out of the 6 DOF) that we have learned.
    """
    start_time = time.time()

    net.eval()

    # Object to keep track of test errors.
    stats = Statistics()

    errors = None
    # We want to print out the avg./max target pose absolute values for reference together with the pose errors.
    targets_total = torch.zeros(6)
    targets_max = torch.zeros(6)
    batch_size = -1
    num_batches, num_examples = 0, 0

    with torch.no_grad():
        for images, disparities, ids, pose_se3, pose_log in data_loader:

            batch_size = images.size(0)

            try:
                # Compute the output poses (we use -1 a placeholder as epoch is not relevant).
                output_se3 = pipeline.forward(net, images, disparities, pose_se3, pose_log, epoch=-1, test=True)

            except Exception as e:
                print(e)
                print("Ids: {}".format(ids))
                continue

            num_batches += 1
            num_examples += batch_size

            # Get the error in each of the 6 pose DOF.
            diff = se3_log(output_se3.inverse().bmm(pose_se3.cuda())).detach().cpu().numpy()
            if num_batches == 1:
                errors = diff
            else:
                errors = np.concatenate((errors, diff), axis=0)

            targets_total += torch.sum(torch.abs(pose_log), dim=0).detach().cpu()
            targets_max = torch.max(targets_max, torch.max(torch.abs(pose_log.detach().cpu()), dim=0)[0])

            # Collect the poses so that we can plot the results later.
            output_log_np = se3_log(output_se3).detach().cpu().numpy()    # Bx6
            target_log_np = se3_log(pose_se3).detach().cpu().numpy()
            output_se3_np = output_se3.detach().cpu().numpy()              # Bx4x4
            target_se3_np = pose_se3.detach().cpu().numpy()
            # kpt_inliers = torch.sum(saved_data['valid_inliers'][(0, 1)], dim=2).detach().cpu().numpy()  # B x 1

            # Loop over each datas sample in the batch.
            for k in range(output_se3_np.shape[0]):
                sample_id = ids[k]
                _, live_run_id, _, map_run_id, _ = re.findall('\w+', sample_id)

                stats.add_sample_id(live_run_id, sample_id)
                stats.add_live_run_id(live_run_id)
                stats.set_map_run_id(map_run_id)
                stats.add_outputs_targets_se3(live_run_id, output_se3_np[k, :, :], target_se3_np[k, :, :])
                stats.add_outputs_targets_log(live_run_id, output_log_np[k, :], target_log_np[k, :])
                # stats.add_kpt_inliers(live_run_id, kpt_inliers[k, 0])

    # Compute errors.
    # RMSE for each pose DOF.
    errors[:, 3:6] = np.rad2deg(errors[:, 3:6])
    test_errors = np.sqrt(np.mean(errors ** 2, axis=0)).reshape(1, 6)

    targets_avg = (targets_total * (1.0 / num_examples)).detach().cpu().numpy()
    targets_avg[3:6] = np.rad2deg(targets_avg[3:6])
    targets_max = targets_max.detach().cpu().numpy()
    targets_max[3:6] = np.rad2deg(targets_max[3:6])

    print(f'\nMap run id: {stats.get_map_run_id()}')
    print(f'Live run ids: {stats.get_live_run_ids()}')
    print(f'Pose RMSE by DOF: {test_errors[:, np.array(dof)]}')
    print(f'Average pose targets by DOF: {targets_avg[np.array(dof)]}')
    print(f'Max pose targets by DOF: {targets_max[np.array(dof)]}')

    print(f'Path test duration: {time.time() - start_time} seconds.\n')

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

    # Find which of the DOF of the 6 DOF pose we want to learn. Record the indices to keep in the pose vector.
    dof = [0, 1, 5] if 'pose_plane' in config['loss']['types'] else [0, 1, 2, 3, 4, 5]

    # The data loaders are stored in a dict that maps from path name to a list of data loaders as we may test using
    # data from more than one path. Also, for each path we localize different runs against each other. One run is used
    # as the map and then one or more other runs are localized to this map run. One individual data loader contains all
    # the samples for all live localized back to one map run.
    for path_name in test_loaders.keys():

        print(f'\nTesting path: {path_name}')

        start_time = time.time()

        # Loop over each data loader (one data loader per map run we localize to).
        for i in range(len(test_loaders[path_name])):

            path_stats = test_model(pipeline, net, test_loaders[path_name][i], dof)

            outputs_log = path_stats.get_outputs_log()
            targets_log = path_stats.get_targets_log()
            map_run_id = path_stats.get_map_run_id()
            live_run_ids = path_stats.get_live_run_ids()
            sample_ids = path_stats.get_sample_ids()

            # Plot each DOF of the estimated and target poses for each pair of live run localized to map run.
            plotting.plot_outputs(outputs_log, targets_log, path_name, map_run_id, dof)

            # Compute the RMSE for translation and rotation if we are using all 6 DOF.
            # TODO: provide the same for SE(2).
            if len(dof) == 6:
                outputs_se3 = path_stats.get_outputs_se3()
                targets_se3 = path_stats.get_targets_se3()
                rmse(outputs_se3, targets_se3)

        print(f'Test time for path {path_name}: {time.time() - start_time}')

def main(config):
    """
        Set up everything needed for training and call the function to run the training loop.

        Args:
            config (dict): configurations for training the network.
    """
    results_path = f"{config['home_path']}/results/{config['experiment_name']}/"
    checkpoints_path = f"{config['home_path']}/networks"
    data_path = f"{config['home_path']}/data"
    datasets_path = f"{config['home_path']}/datasets"

    checkpoint_name = config['checkpoint_name']
    dataset_name = config['dataset_name']

    dataset_path = f'{datasets_path}/{dataset_name}.pickle'
    checkpoint_path = f'{checkpoints_path}/{checkpoint_name}.pth'

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Print outputs to files
    orig_stdout = sys.stdout
    fl = open(f'{results_path}out_test.txt', 'w')
    sys.stdout = fl
    orig_stderr = sys.stderr
    fe = open(f'{results_path}err_test.txt', 'w')
    sys.stderr = fe

    # Record the config settings.
    print(config)

    # Load the data.
    dataloader_params = config['data_loader']
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

    test(testing_pipeline, net, path_loaders, results_path)

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