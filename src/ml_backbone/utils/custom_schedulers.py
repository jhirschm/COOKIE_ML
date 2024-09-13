import warnings
from torch.optim import Optimizer
import torch
'''
Patience defines when to reduce learning rate if had N number of bad epochs
Cool down defines M number of epochs to disregard after updating learning rate if dont get improvement. However, if get improvement leave cool down
Early stop total number of bad epochs to stop training. Dont count epochs during cool down though. Reset if improvement

'''
class CustomScheduler:
    def __init__(self, optimizer, patience=5, early_stop_patience = 8, cooldown=2, lr_reduction_factor=0.1, max_num_epochs = 200, improvement_percentage=0.01):
        # Ensure the optimizer is a valid Optimizer instance
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f'{type(optimizer).__name__} is not an Optimizer')
        self.optimizer = optimizer
        self.patience = patience  # Number of epochs with no improvement after which LR will be reduced
        self.cooldown = cooldown  # Number of epochs to wait before resuming normal operation after LR has been reduced
        self.lr_reduction_factor = lr_reduction_factor  # Factor by which the LR will be reduced
        self.max_num_epochs = max_num_epochs  # Maximum number of epochs to train for
        self.improvement_percentage = improvement_percentage  # Percentage improvement to be considered as significant
        self.best_val_loss = float('inf')  # Initialize the best validation loss to infinity
        self.num_bad_epochs = 0  # Counter for epochs with no improvement
        self.num_bad_epochs_early_stop = 0  # Counter for epochs with no improvement for early stopping
        self.total_epochs = 0  # Counter for total epochs
        self.num_since_lr_reduction = 0  # Counter for epochs since the last LR reduction
        self.lr_reduced = False  # Flag to indicate if LR was reduced in the last epoch
        self.cooldown_counter = 0  # Counter for cooldown epochs
        self.last_epoch = 0  # Initialize last_epoch
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]  # Store initial learning rates
        self.early_stop_patience = early_stop_patience  # Number of epochs with no improvement after which training will be stopped
        if self.early_stop_patience < self.patience:
            warnings.warn("Early stop patience is less than patience. Learning rate reduction may not be triggered.", UserWarning)
        if self.cooldown > self.patience:
            warnings.warn("LR update patience is less than cooldown. Early stop may not be triggered.", UserWarning)
        

    def step(self, val_loss, epoch=None):
        # Update current validation loss and epoch
        current = float(val_loss)
        if epoch is None:
            epoch = self.last_epoch + 1  # Increment epoch if not provided
        else:
            warnings.warn("Epoch parameter is deprecated and will be removed in a future release.", UserWarning)
        self.last_epoch = epoch

        if self.total_epochs >= self.max_num_epochs:
            # If the total number of epochs exceeds the maximum, stop training
            print(f"Maximum number of epochs ({self.max_num_epochs+1}) reached.")
            return True
        elif self.num_bad_epochs_early_stop >= self.early_stop_patience:
            # If bad epochs exceed early_stop_patience, stop training
            print(f"Early stopping at epoch {epoch+1}")
            return True
        

        # Calculate the required improvement in validation loss
        required_improvement = self.best_val_loss * (1 - self.improvement_percentage)
        # print(self.best_val_loss)
        # print(required_improvement)

        if current < required_improvement:
            # print("Improvement")
            # If there is an improvement, update the best validation loss and reset bad epochs counter
            self.best_val_loss = current
            self.num_bad_epochs = 0
            self.num_bad_epochs_early_stop = 0
            self.lr_reduced = False
            self.cooldown_counter = 0
        else:
            if self.num_bad_epochs >= self.patience:
                # If bad epochs exceed patience, reduce learning rate
                self.reduce_lr(epoch)
                self.cooldown_counter = self.cooldown  # Reset cooldown counter
                self.num_bad_epochs = 0  # Reset bad epochs counter
            else:
                self.num_bad_epochs += 1 #If no improvement and didn't exceed patience, increment bad epochs counter

            
            if not self.in_cooldown: # If no improvement and not in cooldown, increment early stop counter
                self.num_bad_epochs_early_stop += 1
            else:
                self.cooldown_counter -= 1
        
        # Update last learning rates
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

        return False
    def reduce_lr(self, epoch):
        # Reduce the learning rate for each parameter group
        for param_group in self.optimizer.param_groups:
            old_lr = float(param_group['lr'])
            new_lr = old_lr * self.lr_reduction_factor 
            param_group['lr'] = new_lr  # Update learning rate
            print(f"Epoch {epoch+1}: reducing learning rate from {old_lr:.6f} to {new_lr:.6f}") # Assuming epoch starts from 0

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

    # def should_stop(self):
    #     # Check if training should stop based on bad epochs and cooldown
    #     return self.num_bad_epochs >= self.patience and self.num_since_lr_reduction >= self.cooldown


class CustomSchedulerWeightUpdate:
    def __init__(self, optimizer, model, patience=5, early_stop_patience=8, cooldown=2, lr_reduction_factor=0.1, 
                 max_num_epochs=200, improvement_percentage=0.01):
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f'{type(optimizer).__name__} is not an Optimizer')
        self.optimizer = optimizer
        self.model = model  
        self.patience = patience
        self.cooldown = cooldown
        self.lr_reduction_factor = lr_reduction_factor
        self.max_num_epochs = max_num_epochs
        self.improvement_percentage = improvement_percentage
        self.best_val_loss = float('inf')
        self.best_model_weights = None  
        self.num_bad_epochs = 0
        self.num_bad_epochs_early_stop = 0
        self.total_epochs = 0
        self.lr_reduced = False
        self.cooldown_counter = 0
        self.last_epoch = 0
        self.early_stop_patience = early_stop_patience
        # self.evaluate_validation_loss = loss_function  # Function to evaluate validation loss

    def step(self, val_loss, epoch=None):
        current = float(val_loss)
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.total_epochs += 1

        if self.total_epochs >= self.max_num_epochs:
            print(f"Maximum number of epochs ({self.max_num_epochs}) reached.")
            return True  

        if self.num_bad_epochs_early_stop >= self.early_stop_patience:
            print(f"Early stopping at epoch {epoch+1}")
            return True  

        required_improvement = self.best_val_loss * (1 - self.improvement_percentage)

        if current < required_improvement:
            self.best_val_loss = current
            self.best_model_weights = self.model.state_dict()  
            self.num_bad_epochs = 0
            self.num_bad_epochs_early_stop = 0
            self.lr_reduced = False
            self.cooldown_counter = 0
        else:
            if self.num_bad_epochs >= self.patience:
                self.reduce_lr(epoch)
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1

            if not self.in_cooldown:
                self.num_bad_epochs_early_stop += 1  
            else:
                self.cooldown_counter -= 1  

        return False  

    def reduce_lr(self, epoch):
        if self.best_model_weights is not None:
            print(f"Loading best model weights before reducing learning rate at epoch {epoch+1}")
            self.model.load_state_dict(self.best_model_weights)
            self.reset_optimizer()  # Reset optimizer state

        for param_group in self.optimizer.param_groups:
            old_lr = float(param_group['lr'])
            new_lr = old_lr * self.lr_reduction_factor 
            param_group['lr'] = new_lr
            print(f"Epoch {epoch+1}: reducing learning rate from {old_lr:.6f} to {new_lr:.6f}")

        # self.verify_model(epoch)  # Double-check if the model improves after LR reduction

    def reset_optimizer(self):
        """Resets the optimizer's state to ensure that momentum, etc., is cleared after loading weights."""
        self.optimizer.state = {}

    # def verify_model(self, epoch):
    #     """Run a validation step after loading weights and reducing LR to verify improvement."""
    #     self.model.eval()
    #     with torch.no_grad():
    #         val_loss = self.evaluate_validation_loss()  # Assume you have a method to calculate val_loss

    #     print(f"Validation loss after LR reduction at epoch {epoch+1}: {val_loss}")
    #     if val_loss > self.best_val_loss:
    #         print("Validation loss increased after LR reduction, considering early stopping or further LR reduction.")

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def state_dict(self):
        return {
            'best_val_loss': self.best_val_loss,
            'num_bad_epochs': self.num_bad_epochs,
            'num_bad_epochs_early_stop': self.num_bad_epochs_early_stop,
            'cooldown_counter': self.cooldown_counter,
            'last_epoch': self.last_epoch,
        }

    def load_state_dict(self, state_dict):
        self.best_val_loss = state_dict['best_val_loss']
        self.num_bad_epochs = state_dict['num_bad_epochs']
        self.num_bad_epochs_early_stop = state_dict['num_bad_epochs_early_stop']
        self.cooldown_counter = state_dict['cooldown_counter']
        self.last_epoch = state_dict['last_epoch']

