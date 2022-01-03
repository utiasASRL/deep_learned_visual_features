import numpy as np
import torch

class EarlyStopping:
    """Based on https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
       Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=7, loss_min=10e9, epoch_min):
        """
        Initialize by setting the chosen patience and set the minimum loss.

        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
        """
        self.patience = patience
        self.counter = 0
        self.loss_min = loss_min
        self.epoch_min = epoch_min

    def check_stop(self, loss, model, optimizer, model_file_path, train_stats, validation_stats, epoch):
        """
            Check if model training needs to stop based on the validation loss. Save the model of the validation loss
            improves (or stays at minimum).

            Args:
                loss (float): validation loss.
                model (torch.nn.Module): neural network model.
                optimizer (torch.optim.Optimizer): training optimizer.
                model_file_path (string name): file path for saving the model.
                train_stats (float): validation loss. TODO
                validation_stats (float): validation loss. TODO
                epoch (int): training epoch.
       """

        if loss <= self.loss_min:

            print(f'Validation loss decreased ({self.loss_min:.6f} --> {loss:.6f}). Saving model at epoch {epoch}.\n')

            self.counter = 0
            self.loss_min = loss
            self.epoch_min = epoch

            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                        'train_stats': train_stats,
                        'valid_stats': validation_stats,
                        }, '{}_{}.pth'.format(model_file_path, epoch))

        elif loss > self.loss_min:

            self.counter += 1

            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            print(f'Current validation loss: {loss:.10f}, min validation loss: {self.loss_min:.10f}\n')

            if self.counter >= self.patience:
                print('Early stopping: final model saved at epoch {}'.format(self.epoch_min))
                return True, self.loss_min

        return False, self.loss_min