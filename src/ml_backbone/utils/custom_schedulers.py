class CustomScheduler:
    def __init__(self, optimizer, patience=5, cooldown=2, lr_reduction_factor=0.1, min_lr=1e-6, improvement_percentage=0.01):
        self.optimizer = optimizer
        self.patience = patience
        self.cooldown = cooldown
        self.lr_reduction_factor = lr_reduction_factor
        self.min_lr = min_lr
        self.improvement_percentage = improvement_percentage
        self.best_val_loss = float('inf')
        self.num_bad_epochs = 0
        self.num_since_lr_reduction = 0
        self.lr_reduced = False

    def step(self, val_loss):
        # Calculate the required improvement as a percentage of the best validation loss
        required_improvement = self.best_val_loss * (1 - self.improvement_percentage)

        if val_loss < required_improvement:
            self.best_val_loss = val_loss
            self.num_bad_epochs = 0
            self.lr_reduced = False
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            self.reduce_lr()
            self.num_bad_epochs = 0
            self.lr_reduced = True
            self.num_since_lr_reduction = 0

        if self.lr_reduced:
            self.num_since_lr_reduction += 1

    def reduce_lr(self):
        for param_group in self.optimizer.param_groups:
            new_lr = max(param_group['lr'] * self.lr_reduction_factor, self.min_lr)
            param_group['lr'] = new_lr

    def should_stop(self):
        return self.num_bad_epochs >= self.patience and self.num_since_lr_reduction >= self.cooldown
