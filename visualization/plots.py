import time
import sys
import os

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

class Plotting:
    """
        Class for plotting results.
    """

    def __init__(self, results_dir):
        """
            Initialize plotting.

            Args:
                results_dir (string): the directory in which to store the plots.
        """
        self.results_dir = results_dir

    def plot_epoch_losses(self, epoch_losses_train, epoch_losses_valid):
        """
            Plot the average training and validation loss for each epoch. Plot each individual type of loss and also
            the weighted sum of the losses.

            Args:
                epoch_losses_train (dict): the average training losses for each epoch.
                epoch_losses_valid (dict): the average training losses for each epoch.
        """
        for loss_type in epoch_losses_train.keys():

            plt.figure()
            p1 = plt.plot(epoch_losses_train[loss_type])
            p2 = plt.plot(epoch_losses_valid[loss_type])
        
            plt.legend((p1[0], p2[0]), ('training', 'validation'))
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.title(f'Loss for each epoch, {loss_type}')
        
            plt.savefig(f'{self.results_dir}loss_epoch_{loss_type}.png', format='png')
            plt.savefig(f'{self.results_dir}loss_epoch_{loss_type}.pdf', format='pdf')
            plt.close()

            plt.figure()
            p1 = plt.plot(np.log(epoch_losses_train[loss_type]))
            p2 = plt.plot(np.log(epoch_losses_valid[loss_type]))

            plt.legend((p1[0], p2[0]), ('training', 'validation'))
            plt.ylabel('Log of loss')
            plt.xlabel('Epoch')
            plt.title('Log of loss for each epoch, {}'.format(loss_type))

            plt.savefig(f'{self.results_dir}log_loss_epoch_{loss_type}.png', format='png')
            plt.savefig(f'{self.results_dir}log_loss_epoch_{loss_type}.pdf', format='pdf')
            plt.close()

    def plot_epoch_errors(self, epoch_error_train, epoch_error_valid, dof):
        """
            Plot the average error for each specified pose DOF for each epoch for training and validation.

            Args:
                epoch_error_train (dict): the average pose errors for each DOF for each epoch.
                epoch_error_valid (dict): the average pose errors for each DOF for each epoch.
                dof (list[int]): indices of the DOF to plot.
        """
        dof_str = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']

        for i in range(len(dof)):
            dof_index = dof[i]

            plt.figure()
            p1 = plt.plot(epoch_error_train[:, dof_index])
            p2 = plt.plot(epoch_error_valid[:, dof_index])

            plt.legend((p1[0], p2[0]), ('training', 'validation'))
            plt.ylabel('RMSE')
            plt.xlabel('Epoch')
            plt.title(f'Error for each epoch, {dof_str[dof_index]}')

            plt.savefig(f'{self.results_dir}error_epoch_{dof_str[dof_index]}.png', format='png')
            plt.savefig(f'{self.results_dir}error_epoch_{dof_str[dof_index]}.pdf', format='pdf')
            plt.close()

    def plot_outputs(self, outputs, outputs_tr, targets, targets_tr, method, epoch, base_id, path_type, ylim=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]):
        """
            Plot the average error for each specified pose DOF for each epoch for training and validation.

            Args:
                epoch_error_train (dict): the average pose errors for each DOF for each epoch.
                epoch_error_valid (dict): the average pose errors for each DOF for each epoch.
                dof (list[int]): indices of the DOF to plot.
        """

        dof = ['x', 'y', 'z', 'rot_x', 'rot_y', 'rot_z'] if outputs[list(outputs)[0]].shape[1] == 6 else ['x', 'y', 'theta']

        for run in outputs.keys():

            directory = self.results_dir + method + '/' + path_type +  '/run_' + str(run)
            if not os.path.exists(directory):
                os.makedirs(directory)

            for j in range(len(dof)):

                targets_plot = np.rad2deg(targets[run][:, j]) if (dof[j] in ['theta', 'rot_x', 'rot_y', 'rot_z']) else np.stack(targets_tr[run], axis=0)[:, j, 3]
                outputs_plot = np.rad2deg(outputs[run][:, j]) if (dof[j] in ['theta', 'rot_x', 'rot_y', 'rot_z']) else np.stack(outputs_tr[run], axis=0)[:, j, 3]

                f = plt.figure(figsize=(18,6))
                f.tight_layout(rect=[0, 0.03, 1, 0.95])
                p1 = plt.plot(targets_plot, 'C1')
                p2 = plt.plot(outputs_plot, 'C0')
                plt.legend((p1[0], p2[0]), ('target', 'predicted'))
                if ylim[j] > 0.0:
                    plt.ylim([-ylim[j], ylim[j]])
                plt.ylabel(dof[j])
                plt.xlabel('Vertex')
                plt.title(dof[j] + ' - ' + method)
                #plt.savefig(directory + '/' + dof[j] + '_base_' + str(base_ids[run])  + '_epoch_' + str(epoch) + '.png', format='png')
                plt.savefig(directory + '/' + dof[j] + '_base_' + str(base_id)  + '.png', format='png')
                plt.close()

                f = plt.figure(figsize=(18,6))
                f.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.plot(targets_plot - outputs_plot, 'C3')
                if ylim[j] > 0.0:
                    plt.ylim([-ylim[j], ylim[j]])
                plt.ylabel('Difference in ' + dof[j])
                plt.xlabel('Vertex')
                plt.title(dof[j] + ' - ' + method)
                #plt.savefig(directory + '/' + dof[j] + '_diff_base_' + str(base_ids[run])  + '_epoch_' + str(epoch) + '.png', format='png')
                plt.savefig(directory + '/' + dof[j] + '_diff_base_' + str(base_id) + '.png', format='png')
                plt.close()


    def plot_cumulative_err(self, errors, error_type, unit, path_name, run_id, base_id):

        directory = self.results_dir + path_name + '/' + path_type + '/run_' + str(run_id)
        xlim = [0.0, 1.0] if error_type == 'translation' else [0.0, 3.0]

        # Put all the values that go over xlim in one bucket (so size of buckets don't stretch with large max values)
        # max_vals = np.ones(errors.shape) * (xlim[1] + 0.01)
        # indices = errors > xlim[1]
        # errors[indices] = max_vals[indices]

        max_val = np.max(errors)
        n_bins_vis_range = 50
        n_bins_total = int((n_bins_vis_range * max_val) / xlim[1])

        if not os.path.exists(directory):
            os.makedirs(directory)

        f = plt.figure(figsize=(20, 10))
        f.tight_layout(rect=[0, 0.03, 1, 0.95])
        values, base = np.histogram(errors, bins=n_bins_total)
        unity_values = values / values.sum()
        cumulative = np.cumsum(unity_values)
        plt.plot(base[:-1], cumulative)
        plt.xlim(xlim)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlabel('Error in ' + error_type + ' (' + unit + ')', fontsize=20, weight='bold')
        plt.ylabel('Cumulative distribution', fontsize=20, weight='bold')
        plt.savefig(directory + '/err_hist_' + error_type + '_base_' + base_id + '.png', format='png')
        plt.close()