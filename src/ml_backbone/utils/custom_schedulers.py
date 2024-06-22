import warnings
from torch.optim import Optimizer

class CustomScheduler:
    def __init__(self, optimizer, patience=5, cooldown=2, lr_reduction_factor=0.1, min_lr=1e-6, improvement_percentage=0.01):
        # Ensure the optimizer is a valid Optimizer instance
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f'{type(optimizer).__name__} is not an Optimizer')
        self.optimizer = optimizer
        self.patience = patience  # Number of epochs with no improvement after which LR will be reduced
        self.cooldown = cooldown  # Number of epochs to wait before resuming normal operation after LR has been reduced
        self.lr_reduction_factor = lr_reduction_factor  # Factor by which the LR will be reduced
        self.min_lr = min_lr  # Minimum learning rate
        self.improvement_percentage = improvement_percentage  # Percentage improvement to be considered as significant
        self.best_val_loss = float('inf')  # Initialize the best validation loss to infinity
        self.num_bad_epochs = 0  # Counter for epochs with no improvement
        self.num_since_lr_reduction = 0  # Counter for epochs since the last LR reduction
        self.lr_reduced = False  # Flag to indicate if LR was reduced in the last epoch
        self.cooldown_counter = 0  # Counter for cooldown epochs
        self.last_epoch = 0  # Initialize last_epoch
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]  # Store initial learning rates

    def step(self, val_loss, epoch=None):
        # Update current validation loss and epoch
        current = float(val_loss)
        if epoch is None:
            epoch = self.last_epoch + 1  # Increment epoch if not provided
        else:
            warnings.warn("Epoch parameter is deprecated and will be removed in a future release.", UserWarning)
        self.last_epoch = epoch

        # Calculate the required improvement in validation loss
        required_improvement = self.best_val_loss * (1 - self.improvement_percentage)

        if current < required_improvement:
            # If there is an improvement, update the best validation loss and reset bad epochs counter
            self.best_val_loss = current
            self.num_bad_epochs = 0
            self.lr_reduced = False
        else:
            # If no improvement, increment bad epochs counter
            self.num_bad_epochs += 1

        if self.in_cooldown:
            # If in cooldown period, decrement cooldown counter and reset bad epochs counter
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0

        if self.num_bad_epochs > self.patience:
            # If bad epochs exceed patience, reduce learning rate
            self.reduce_lr(epoch)
            self.cooldown_counter = self.cooldown  # Reset cooldown counter
            self.num_bad_epochs = 0  # Reset bad epochs counter

        # Update last learning rates
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def reduce_lr(self, epoch):
        # Reduce the learning rate for each parameter group
        for param_group in self.optimizer.param_groups:
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.lr_reduction_factor, self.min_lr)  # Ensure new_lr is not less than min_lr
            param_group['lr'] = new_lr  # Update learning rate

    @property
    def in_cooldown(self):
        # Check if the scheduler is in the cooldown period
        return self.cooldown_counter > 0

    def state_dict(self):
        # Return the state of the scheduler for saving
        return {
            'best_val_loss': self.best_val_loss,
            'num_bad_epochs': self.num_bad_epochs,
            'num_since_lr_reduction': self.num_since_lr_reduction,
            'lr_reduced': self.lr_reduced,
            'cooldown_counter': self.cooldown_counter,
            'last_epoch': self.last_epoch,
            '_last_lr': self._last_lr
        }

    def load_state_dict(self, state_dict):
        # Load the scheduler state from a saved state_dict
        self.best_val_loss = state_dict['best_val_loss']
        self.num_bad_epochs = state_dict['num_bad_epochs']
        self.num_since_lr_reduction = state_dict['num_since_lr_reduction']
        self.lr_reduced = state_dict['lr_reduced']
        self.cooldown_counter = state_dict['cooldown_counter']
        self.last_epoch = state_dict['last_epoch']
        self._last_lr = state_dict['_last_lr']

    def should_stop(self):
        # Check if training should stop based on bad epochs and cooldown
        return self.num_bad_epochs >= self.patience and self.num_since_lr_reduction >= self.cooldown
