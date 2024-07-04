from denoising_util import *
from typing import List, Any

class StepFunction(nn.Module):
    def __init__(self, threshold: float = 0.5):
        super(StepFunction, self).__init__()
        self.threshold = threshold
    
    def forward(self, x):
        return (x > self.threshold).float()

class Zero_PulseClassifier(nn.Module):
    def __init__(self, threshold: float = 0.5):
        super(Zero_PulseClassifier, self).__init__()
        self.linear = nn.Linear(1, 1)
        self.step_function = StepFunction(threshold)

    def forward(self, x):
        x = self.linear(x)
        x = self.step_function(x)
        return x

    # def predict(self, img):
    #     # Sum all points in the image over the image dimensions (dim=2 and dim=3)
    #     sum_of_points = torch.sum(img, dim=(2, 3), keepdim=True)
    #     # Flatten the sum to (batch_size, 1)
    #     sum_of_points = sum_of_points.view(sum_of_points.size(0), -1)
    #     return self.forward(sum_of_points)
    
    def train_model(self, train_dataloader, val_dataloader, criterion, optimizer, scheduler, model_save_dir, identifier, device, checkpoints_enabled=True, resume_from_checkpoint=False, max_epochs=10):
        self.to(device)
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

        name = f"{model_save_dir}/{identifier}"+"_run_time_info.txt"
    
        with open(name, "a") as f:
            f.write(f"Training resumed at {datetime.datetime.now()} from epoch {start_epoch}\n" if start_epoch > 0 else f"Training started at {datetime.datetime.now()}\n")

            for epoch in range(start_epoch, max_epochs):
                self.train()  # Set the model to training mode
                running_train_loss = 0.0

                for batch in train_dataloader:
                    optimizer.zero_grad()  # Zero the parameter gradients

                    inputs, labels = batch
                    print(inputs)
                    print(labels)
                    inputs = torch.unsqueeze(inputs, 1)
                    inputs = inputs.to(device, torch.float32)
                    labels = labels.to(device, torch.float32)
                    sum_of_points = torch.sum(inputs, dim=(2, 3), keepdim=True)
                    # Flatten the sum to (batch_size, 1)
                    sum_of_points = sum_of_points.view(sum_of_points.size(0), -1)
                    sum_of_points = sum_of_points.to(device, torch.float32)

                    outputs = self(sum_of_points).to(device)

                

                    labels = labels[:,1:].to(device)
                    # Ensure outputs and labels require grad
                    if not outputs.requires_grad:
                        outputs.requires_grad_(True)
                    if not labels.requires_grad:
                        labels.requires_grad_(True)

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
                        inputs = torch.unsqueeze(inputs, 1)
                        inputs = inputs.to(device, torch.float32)
                        labels = labels.to(device, torch.float32)

                        sum_of_points = torch.sum(inputs, dim=(2, 3), keepdim=True)
                        # Flatten the sum to (batch_size, 1)
                        sum_of_points = sum_of_points.view(sum_of_points.size(0), -1)
                        sum_of_points = sum_of_points.to(device, torch.float32)


                        outputs = self(sum_of_points).to(device)
                        #only use second element
                        labels = labels[:,1:].to(device)
                        loss = criterion(outputs, labels)
                        running_val_loss += loss.item()

                val_loss = running_val_loss / len(val_dataloader)
                val_losses.append(val_loss)

                f.write(f"Epoch [{epoch+1}/{max_epochs}] - Train Loss: {train_loss:.10f}, Validation Loss: {val_loss:.10f}\n\n")
                print(f"Epoch [{epoch+1}/{max_epochs}] - Train Loss: {train_loss:.10f}, Validation Loss: {val_loss:.10f}\n\n")

                # Update the scheduler
                should_stop = scheduler.step(val_loss, epoch)

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
                if should_stop:
                    print(f"Early stopping at epoch {epoch+1}\n")
                    f.write(f"Early stopping at epoch {epoch+1}\n")
                    break
                f.flush() # Flush the buffer to write to the file
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

 
class Ximg_to_Ypdf_Autoencoder(nn.Module):
    def __init__(self, encoder_layers: List[List[Any]], decoder_layers: List[List[Any]], dtype=torch.float32):
        super(Ximg_to_Ypdf_Autoencoder, self).__init__()
        self.dtype = dtype
        
        # Create encoder based on the provided layer configuration
        encoder_modules = []
        
        for i in range(encoder_layers.shape[0]):
            layer = encoder_layers[i,0]
            activation = encoder_layers[i,1]
            encoder_modules.append(layer)
            if activation is not None:
                encoder_modules.append(activation)
        
        self.encoder = nn.Sequential(*encoder_modules)
        # Cast encoder weights to torch.float32
        for param in self.encoder.parameters():
            param.data = param.data.to(self.dtype)
        
        # Create decoder based on the provided layer configuration
        decoder_modules = []
        for i in range(len(decoder_layers)):
            layer = decoder_layers[i,0]
            activation = decoder_layers[i,1]
            decoder_modules.append(layer)
            if activation is not None:
                decoder_modules.append(activation)
        
        self.decoder = nn.Sequential(*decoder_modules)
        for param in self.decoder.parameters():
            param.data = param.data.to(self.dtype)

        # Side network for binary classification
        self.side_network = nn.Sequential(
            nn.Linear(1, 1),
            StepFunction(threshold=0.5)  # Step function for binary output
        )

    def forward(self, x):
        # Side network forward pass
        sum_of_points = torch.sum(x, dim=(2, 3), keepdim=True)  # Sum over img_dim_x and img_dim_y
        sum_of_points = sum_of_points.view(sum_of_points.size(0), -1)  # Flatten to (batch_size, 1)
        side_output = self.side_network(sum_of_points)
        side_output = side_output.view(-1, 1, 1, 1) 

        x = self.encoder(x)
        x = self.decoder(x)

        return x
    
    def freeze_all_layers(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_layers(self, encoder_layer_indices, decoder_layer_indices):
        for idx in encoder_layer_indices:
            for param in self.encoder[idx].parameters():
                param.requires_grad = True
        
        for idx in decoder_layer_indices:
            for param in self.decoder[idx].parameters():
                param.requires_grad = True
    

    def train_model(self, train_dataloader, val_dataloader, criterion, optimizer, scheduler, model_save_dir, identifier, device, checkpoints_enabled=True, resume_from_checkpoint=False, max_epochs=10):
        self.to(device)
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
                    inputs = torch.unsqueeze(inputs, 1)
                    inputs = inputs.to(device, torch.float32)
                    
                    

                    outputs = self(inputs).to(device)
                    outputs = outputs.squeeze()  # Remove channel dimension
                    
                    labels = labels.squeeze()
                    labels = labels.to(device)
                    
                    # Set a small threshold value
                    threshold = 1e-6

                    # Create a mask for labels that are effectively zero
                    zero_mask = (torch.sum(labels ** 2, dim=(1, 2)) < threshold).float()  # Assuming labels are 2D images in the batch

                    # Calculate the loss for each sample in the batch
                    losses = criterion(outputs, labels)
                    
                    # Apply the weighting for zero labels
                    # weighted_losses = losses * (1 + zero_mask * 9)  # Increase loss by a factor of 10 for zero labels
                    weighted_losses = losses * (1 + zero_mask * 19)  # Increase loss by a factor of 20 for zero labels

                    
                    # Compute the mean loss
                    loss = torch.mean(weighted_losses)

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
                        inputs = torch.unsqueeze(inputs, 1)
                        inputs = inputs.to(device, torch.float32)

                        
                        outputs = self(inputs).to(device)
                        outputs = outputs.squeeze()

                        labels = labels.squeeze()
                        labels = labels.to(device)
                        loss = criterion(outputs, labels)
                        running_val_loss += loss.item()

                val_loss = running_val_loss / len(val_dataloader)
                val_losses.append(val_loss)

                f.write(f"Epoch [{epoch+1}/{max_epochs}] - Train Loss: {train_loss:.10f}, Validation Loss: {val_loss:.10f}\n\n")
                print(f"Epoch [{epoch+1}/{max_epochs}] - Train Loss: {train_loss:.10f}, Validation Loss: {val_loss:.10f}\n\n")

                # Update the scheduler
                should_stop = scheduler.step(val_loss, epoch)

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
                # if scheduler.should_stop():
                #     print(f"Early stopping at epoch {epoch+1}")
                #     break
                if should_stop:
                    print(f"Early stopping at epoch {epoch+1}\n")
                    f.write(f"Early stopping at epoch {epoch+1}\n")
                    break
                f.flush() # Flush the buffer to write to the file
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

    def evaluate_model(self, dataloader, criterion, device, save_results=False, results_dir=None, results_filename=None):
        # Calcualte the loss on the provided dataloader and save results if specified to H5 file including the input, output, and target
        self.eval()
        running_loss = 0.0
        results = {}


        with torch.no_grad():
            i = 0
            for batch in dataloader:
                inputs, labels = batch
                inputs = torch.unsqueeze(inputs, 1)
                inputs = inputs.to(device, torch.float32)
                # labels = labels[0]
                labels = labels.to(device,torch.float32) #indexing for access to the first element of the list
                outputs = self(inputs)
                outputs = outputs.squeeze()
                outputs = outputs.to(device)
                labels = labels.squeeze()
                loss = criterion(outputs, labels)
                running_loss += loss.item()

                if save_results:
                    # Convert tensors to numpy arrays
                    inputs_np = inputs.cpu().numpy()
                    outputs_np = outputs.cpu().numpy()
                    labels_np = labels.cpu().numpy()
                    results[i] = (inputs_np, outputs_np, labels_np, loss.item())
                i+=1

        avg_loss = running_loss / len(dataloader)

        if save_results and results_dir and results_filename:
            results_filepath = f"{results_dir}/{results_filename}"
            with h5py.File(results_filepath, 'w') as h5file:
                for batch_idx, (inputs_np, outputs_np, labels_np, loss) in results.items():
                    for example_idx in range(inputs_np.shape[0]):
                        group = h5file.create_group(f"{batch_idx}_{example_idx}")
                        group.create_dataset('input', data=inputs_np[example_idx].reshape(16, 512))
                        group.create_dataset('output', data=outputs_np[example_idx].reshape(16, 512))
                        group.create_dataset('target', data=labels_np[example_idx].reshape(16, 512))
                        group.attrs['loss'] = loss  # Store the loss as an attribute
                        

        return avg_loss

    def fine_tune(self, train_dataloader, val_dataloader, criterion, optimizer, scheduler, model_save_dir, identifier, device, encoder_layer_indices_unfreeze, decoder_layer_indices_unfreeze, initial_weights_path, max_epochs=10, gradient_clipping_value=0.01, learning_rate_scale=0.1):
        self.to(device)
        self.load_state_dict(torch.load(initial_weights_path, map_location=device))
    
        # Freeze all layers
        self.freeze_all_layers()

    
        self.unfreeze_layers(encoder_layer_indices_unfreeze, decoder_layer_indices_unfreeze)
        self.to(device)

        
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=optimizer.param_groups[0]['lr'] * learning_rate_scale)
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_epoch = 0
        start_epoch = 0

        name = f"{model_save_dir}/{identifier}"+"_fine_tuning_info.txt"
        with open(name, "a") as f:
            f.write(f"Fine-tuning started at {datetime.datetime.now()}\n")

            for epoch in range(max_epochs):
                self.train()  # Set the model to training mode
                running_train_loss = 0.0

                for batch in train_dataloader:
                    optimizer.zero_grad()  # Zero the parameter gradients

                    inputs, labels = batch
                    inputs = torch.unsqueeze(inputs, 1)
                    inputs = inputs.to(device, torch.float32)
                    
                    outputs = self(inputs).to(device)
                    outputs = outputs.squeeze()  # Remove channel dimension
                    
                    labels = labels.squeeze()
                    labels = labels.to(device)
                    loss = criterion(outputs, labels)

                    # Regularization term
                    reg_term = 0
                    for name, param in self.named_parameters():
                        if param.requires_grad:
                            initial_param = torch.load(initial_weights_path)[name]
                            reg_term += torch.sum((param - initial_param) ** 2)
                    reg_term = 1e-3 * reg_term  # Regularization factor
                    loss += reg_term

                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.parameters(), gradient_clipping_value)
                    
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
                        inputs = torch.unsqueeze(inputs, 1)
                        inputs = inputs.to(device, torch.float32)

                        outputs = self(inputs).to(device)
                        outputs = outputs.squeeze()

                        labels = labels.squeeze()
                        labels = labels.to(device)
                        loss = criterion(outputs, labels)
                        running_val_loss += loss.item()

                val_loss = running_val_loss / len(val_dataloader)
                val_losses.append(val_loss)

                f.write(f"Epoch [{epoch+1}/{max_epochs}] - Train Loss: {train_loss:.10f}, Validation Loss: {val_loss:.10f}\n\n")
                print(f"Epoch [{epoch+1}/{max_epochs}] - Train Loss: {train_loss:.10f}, Validation Loss: {val_loss:.10f}\n\n")

                # Update the scheduler
                should_stop = scheduler.step(val_loss, epoch)

                # Check if this is the best model so far
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    best_model = self.state_dict().copy()

                    # Save the best model with a specified name and path in the model_dir
                    best_model_path = f"{model_save_dir}/{identifier}_fine_tuned_best_model.pth"
                    torch.save(self.state_dict(), best_model_path)

                f.flush()  # Flush the buffer to write to the file
                
                if should_stop:
                    print(f"Early stopping at epoch {epoch+1}\n")
                    f.write(f"Early stopping at epoch {epoch+1}\n")
                    break

        # Save the output to the specified file
        run_summary_path = f"{model_save_dir}/{identifier}" + "_fine_tuning_summary.txt"
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
        losses_path = os.path.join(model_save_dir, identifier + "_fine_tuning_losses.pdf")
        plt.savefig(losses_path)
        plt.close()

        return best_model, best_epoch, train_losses[-1], val_losses[-1], best_val_loss

def main():


    # Example usage
    encoder_layers = [
        (nn.Conv2d(1, 16, kernel_size=3, padding=2), nn.ReLU()),
        (nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU()),
        (nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU()),
    ]

    decoder_layers = [
        (nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1), nn.ReLU()),
        (nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1), nn.ReLU()),
        (nn.ConvTranspose2d(16, 1, kernel_size=3, padding=2), nn.Sigmoid()),  # Example with Sigmoid activation
        # (nn.ConvTranspose2d(16, 1, kernel_size=3, padding=2), None),  # Example without activation
    ]


    autoencoder = Ximg_to_Ypdf_Autoencoder(encoder_layers, decoder_layers)
if __name__ == "__main__":
  

    main()