from denoising_util import *
import DataMilking
class Ximg_to_Ypdf_Autoencoder(nn.Module):
    def __init__(self, encoder_layers, decoder_layers):
        super(Ximg_to_Ypdf_Autoencoder, self).__init__()
        
        # Create encoder based on the provided layer configuration
        encoder_modules = []
        for layer in encoder_layers:
            encoder_modules.append(layer)
            if isinstance(layer, nn.Conv2d):
                encoder_modules.append(nn.ReLU())
        
        self.encoder = nn.Sequential(*encoder_modules)
        
        # Create decoder based on the provided layer configuration
        decoder_modules = []
        for layer in decoder_layers:
            decoder_modules.append(layer)
            if isinstance(layer, nn.ConvTranspose2d):
                decoder_modules.append(nn.ReLU())
        
        # Replace the last ReLU with Sigmoid for the final layer
        if isinstance(decoder_modules[-1], nn.ReLU):
            decoder_modules[-1] = nn.Sigmoid()
        
        self.decoder = nn.Sequential(*decoder_modules)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def train_model(self, train_dataloader, val_dataloader, criterion, optimizer, scheduler, model_save_dir, identifier, device, checkpoints_enabled=True, resume_from_checkpoint=False, max_epochs=10):
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_epoch = 0
        start_epoch = 0

        checkpoint_path = os.path.join(model_save_dir, f"{identifier}_checkpoint.pth")

        # Try to load from checkpoint if it exists and resume_from_checkpoint is True
        if checkpoints_enabled and resume_from_checkpoint and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            train_losses = checkpoint['train_losses']
            val_losses = checkpoint['val_losses']
            best_val_loss = checkpoint['best_val_loss']
            best_epoch = checkpoint['best_epoch']
            best_model = None

        name = f"{model_save_dir}/{identifier}"+"_run_time_info.txt"
        with open(name, "a") as f:
            f.write(f"Training resumed at {datetime.datetime.now()} from epoch {start_epoch}\n" if start_epoch > 0 else f"Training started at {datetime.datetime.now()}\n")


            for epoch in range(max_epochs):
                self.train()  # Set the model to training mode
                running_train_loss = 0.0

                for batch in train_dataloader:
                    optimizer.zero_grad()  # Zero the parameter gradients

                    inputs, labels = batch
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = self(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_train_loss += loss.item()

                train_loss = running_train_loss / len(train_dataloader)
                train_losses.append(train_loss)

                # Validation loop
                self.eval()  # Set the model to evaluation mode
                running_val_loss = 0.0
                
                with torch.no_grad():
                    for batch in val_dataloader:
                        inputs, labels = batch
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = self(inputs)
                        loss = criterion(outputs, labels)
                        running_val_loss += loss.item()

                val_loss = running_val_loss / len(val_dataloader)
                val_losses.append(val_loss)

                print(f"Epoch [{epoch+1}/{max_epochs}] - Train Loss: {train_loss:.10f}, Validation Loss: {val_loss:.10f}")

                # Update the scheduler
                scheduler.step(val_loss)

                # Check if this is the best model so far
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    best_model = self.state_dict().copy()

                    # Save the best model with a specified name and path in the model_dir
                    best_model_path = f"{model_save_dir}/{identifier}_best_model.pth"
                    torch.save(self.state_dict(), best_model_path)

                # Save checkpoint
                if checkpoints_enabled:
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': self.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'train_losses': train_losses,
                        'val_losses': val_losses,
                        'best_val_loss': best_val_loss,
                        'best_epoch': best_epoch,
                    }
                    torch.save(checkpoint, checkpoint_path)
                
                # Early stopping check
                if scheduler.should_stop():
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        # Save the output to the specified file
        run_summary_path = f"{model_save_dir}/{identifier}"+ "_run_summary.txt"
        with open(run_summary_path, "w") as file:
            file.write("Number of Epochs for Best Model: {}\n".format(best_epoch + 1))
            file.write("Final Training Loss: {:.10f}\n".format(train_losses[-1]))
            file.write("Final Validation Loss: {:.10f}\n".format(val_losses[-1]))

        # Plot the training and validation losses
        plt.figure()
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.scatter(best_epoch, val_losses[best_epoch], marker='*', color='red', label='Best Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        losses_path = os.path.join(model_save_dir, identifier + "_losses.pdf")
        plt.savefig(losses_path)
        plt.close()

        return best_model, best_epoch, train_losses[-1], val_losses[-1], best_val_loss

        

# Example usage
encoder_layers = [
    nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
    nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
    nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
]

decoder_layers = [
    nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
    nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
    nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
]

autoencoder = Ximg_to_Ypdf_Autoencoder(encoder_layers, decoder_layers)