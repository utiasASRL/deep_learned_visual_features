import numpy as np

class Statistics:
    """
        A class for keeping track of estimated poses, targets, errors, and losses.
    """
    def __init__(self):
        self.epoch_losses = {}
        self.epoch_errors = np.empty(0)
        self.outputs = {}
        self.targets = {}
        self.live_run_ids = []
        self.map_run_id = None
        self.sample_ids = {}

    def epoch_reset(self):
        """
            Reset the variables tha track the estimated poses and ids for an epoch before a new epoch.
        """
        self.outputs = {}
        self.targets = {}
        self.live_run_ids = []
        self.map_run_id = None
        self.sample_ids = {}

    def get_epoch_stats(self):
        return self.epoch_losses, self.epoch_errors

    def get_outputs(self):
        return self.outputs

    def get_targets(self):
        return self.targets

    def get_sample_ids(self):
        return self.sample_ids

    def get_live_run_ids(self):
        return self.live_run_ids

    def get_map_run_id(self):
        return self.map_run_id

    def add_epoch_stats(self, losses, errors):
        for loss_type in losses.keys():
            if loss_type in self.epoch_losses:
                self.epoch_losses[loss_type] += [losses[loss_type]]
            else:
                self.epoch_losses[loss_type] = [losses[loss_type]]

        if self.epoch_errors.shape[0] == 0:
            self.epoch_errors = errors
        else:
            self.epoch_errors = np.concatenate((self.epoch_errors, errors), axis=0)

    def add_outputs_targets(self, live_run_id, outputs, targets):
        num_dof = outputs.shape[0]

        if live_run_id not in self.outputs.keys():
            self.outputs[live_run_id] = np.zeros((1, num_dof))
            self.targets[live_run_id] = np.zeros((1, num_dof))

        self.outputs[live_run_id] = np.concatenate((self.outputs[live_run_id], outputs.reshape(1, num_dof)), axis=0)
        self.targets[live_run_id] = np.concatenate((self.targets[live_run_id], targets.reshape(1, num_dof)), axis=0)

    def add_outputs_targets_transforms(self, live_run_id, output_T, target_T):
        if live_run_id not in self.outputs_tr.keys():
            self.outputs_tr[live_run_id] = [output_T]
            self.targets_tr[live_run_id] = [target_T]

        self.outputs_tr[live_run_id] = self.outputs_tr[live_run_id] + [output_T]
        self.targets_tr[live_run_id] = self.targets_tr[live_run_id] + [target_T]

    def add_sample_id(self, live_run_id, sample_id):
        if live_run_id not in self.sample_ids.keys():
            self.sample_ids[live_run_id] = [sample_id]
        else:
            self.sample_ids[live_run_id] += [sample_id]

    def add_live_run_id(self, live_run_id):
        if live_run_id not in self.live_run_ids:
            self.live_run_ids.append(live_run_id)

    def set_map_run_id(self, map_run_id):
        self.map_run_id = map_run_id
