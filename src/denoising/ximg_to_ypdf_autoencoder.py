from denoising_util import *
from typing import List, Any

class StepFunction(nn.Module):
    def __init__(self, threshold: float = 0.5):
        super(StepFunction, self).__init__()
        self.threshold = threshold
    
    def forward(self, x):
        return (x > self.threshold).float()

class Zero_PulseClassifier(nn.Module):
    def __init__(self, conv_layers: List[List[Any]], fc_layers: List[List[Any]], dtype=torch.float32):
        super(Zero_PulseClassifier, self).__init__()
        self.dtype = dtype
        
        # Create convolutional layers based on the provided layer configuration
        conv_modules = []
        for layer, activation in conv_layers:
            conv_modules.append(layer)
            if activation is not None:
                conv_modules.append(activation)
        
        self.conv_layers = nn.Sequential(*conv_modules)
        
        # Cast conv layers weights to specified dtype
        for param in self.conv_layers.parameters():
            param.data = param.data.to(self.dtype)
        
        # Create fully connected layers based on the provided layer configuration
        fc_modules = []
        for layer, activation in fc_layers:
            fc_modules.append(layer)
            if activation is not None:
                fc_modules.append(activation)
        
        self.fc_layers = nn.Sequential(*fc_modules)
        
        # Cast fc layers weights to specified dtype
        for param in self.fc_layers.parameters():
            param.data = param.data.to(self.dtype)


    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the output from conv layers
        x = self.fc_layers(x)

        return x

    def predict(self, x):
        """
        Perform binary classification prediction from logits.

        Args:
        - x (torch.Tensor): Input tensor, shape (batch_size, 512)

        Returns:
        - probabilities (torch.Tensor): Probabilities after applying sigmoid activation, shape (batch_size, 1)
        - predictions (torch.Tensor): Binary predictions (0 or 1), shape (batch_size, 1)
        """
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > 0.5).float()
        
        return probabilities, predictions
    
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
                    inputs = torch.unsqueeze(inputs, 1)
                    inputs = inputs.to(device, torch.float32)
                    labels = labels.to(device, torch.float32)
                    

                    outputs = self(inputs).to(device)

                

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

                        # sum_of_points = torch.sum(inputs**2, dim=(2, 3), keepdim=True)
                        # # Flatten the sum to (batch_size, 1)
                        # sum_of_points = sum_of_points.view(sum_of_points.size(0), -1)
                        # sum_of_points = sum_of_points.to(device, torch.float32)


                        outputs = self(inputs).to(device)
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

    def evaluate_model(self, test_dataloader, identifier, model_save_dir, device):
        
        true_pulses = []
        predicted_pulses = []
        
        self.to(device)

        
        self.eval()  # Set the model to evaluation mode, ensures no dropout is applied
        # Iterate through the test data
       

        with torch.no_grad():
            for batch in test_dataloader:
                inputs, labels = batch
                inputs = torch.unsqueeze(inputs, 1)
                inputs = inputs.to(device, torch.float32)
                labels = labels.to(device, torch.float32)

                        
                outputs = self(inputs).to(device)
                # predictions = torch.round(outputs).cpu().numpy()
                # Apply sigmoid to convert logits to probabilities
                probabilities = torch.sigmoid(outputs)

                # Convert probabilities to binary predictions (0 or 1)
                predictions = torch.round(probabilities).cpu().numpy()
                # print(predictions)
                labels = labels[:,1:].to(device)
                # print("labels")
                # print(labels)

                true_pulses.extend(labels.cpu().numpy())
                predicted_pulses.extend(predictions)

        true_pulses = np.array(true_pulses)
        predicted_pulses = np.array(predicted_pulses)
        # print(np.sum(true_pulses))
        # print(np.sum(predicted_pulses))


        num_classes_from_test = 2
        # Calculate evaluation metrics as percentages
        accuracy = accuracy_score(true_pulses, predicted_pulses) * 100
        # print(accuracy)
        # precision = precision_score(true_pulses, predicted_pulses, average='macro') * 100
        # recall = recall_score(true_pulses, predicted_pulses, average='macro') * 100
        # f1 = f1_score(true_pulses, predicted_pulses, average='macro') * 100
        # Confusion matrix
        # cm = confusion_matrix(true_pulses, predicted_pulses)
        # print(true_pulses)
        # print(predicted_pulses)
        # print(cm)
        # # Normalize the confusion matrix based on percentages
        # row_sums = cm.sum(axis=1, keepdims=True)
        # row_sums[row_sums == 0] = 1
        # normalized_cm = cm / row_sums.astype(float) * 100

        # # Create class labels based on the number of classes
        # class_labels = [f'{i} Pulse(s)' for i in range(num_classes_from_test)]

        # # Plot the normalized confusion matrix with class labels
        # plt.figure(figsize=(8, 6))
        # plt.imshow(normalized_cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))

        # # Add class labels to the plot
        # plt.title('Normalized Confusion Matrix (%)')
        # plt.colorbar()
        # tick_marks = np.arange(len(class_labels))
        # plt.xticks(tick_marks, class_labels, rotation=45)
        # plt.yticks(tick_marks, class_labels)

        # for i in range(num_classes_from_test):
        #     for j in range(num_classes_from_test):
        #         text_label = f"{normalized_cm[i, j]:.2f}%"
        #         plt.text(j, i, text_label, horizontalalignment="center", color="black")
        # # Add axis labels
        # plt.xlabel('Predicted')
        # plt.ylabel('True')
        # Calculate the confusion matrix for binary classification
        cm = confusion_matrix(true_pulses, predicted_pulses)
        
        # Print to verify
        print("True Labels:", true_pulses)
        print("Predicted Labels:", predicted_pulses)
        print("Confusion Matrix:\n", cm)

        # Normalize the confusion matrix based on percentages
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Handle any rows that sum to zero
        normalized_cm = cm / row_sums.astype(float) * 100

        # Create class labels for the two classes
        class_labels = ['0 Pulse', '1 Pulse']

        # Plot the normalized confusion matrix with class labels
        plt.figure(figsize=(8, 6))
        plt.imshow(normalized_cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
        plt.title('Normalized Confusion Matrix (%)')
        plt.colorbar()
        tick_marks = np.arange(len(class_labels))
        plt.xticks(tick_marks, class_labels, rotation=45)
        plt.yticks(tick_marks, class_labels)

        for i in range(2):
            for j in range(2):
                text_label = f"{normalized_cm[i, j]:.2f}%"
                plt.text(j, i, text_label, horizontalalignment="center", color="black")
    
        plot_path = os.path.join(model_save_dir, identifier + "_confusion_matrix.pdf")
        # Display the plot
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(plot_path)
        plt.close()

        # Print and display metrics
        # Redirect output to the same log file used during training
        # run_summary_path = f"{model_save_dir}/{identifier}"+ "_run_summary.txt"
        # with open(run_summary_path, "a") as file:
        #     file.write(f"Accuracy: {accuracy:.2f}%\n")
        #     file.write(f"Precision: {precision:.2f}%\n")
        #     file.write(f"Recall: {recall:.2f}%\n")
        #     file.write(f"F1 Score: {f1:.2f}%\n")
        #     for i in range(num_classes_from_test):
        #         true_positives = cm[i, i]  # The diagonal of the confusion matrix
        #         total_instances = np.sum(cm[i, :])  # Total instances in the true class
        #         accuracy_class = true_positives / total_instances * 100

        #         class_name = f"Class {i}"
        #         file.write(f"{class_name} - Accuracy: {accuracy_class:.2f}%\n")

        # return accuracy, precision, recall, f1, normalized_cm, plot_path
class Ximg_to_Ypdf_Autoencoder(nn.Module):
    def __init__(self, encoder_layers: List[List[Any]], decoder_layers: List[List[Any]], dtype=torch.float32, outputEncoder = False):
        super(Ximg_to_Ypdf_Autoencoder, self).__init__()
        self.dtype = dtype
        self.outputEncoder = outputEncoder
        
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

        

    def forward(self, x):
        # Side network forward pass
        
        y = self.encoder(x)
        x = self.decoder(y)

        if self.outputEncoder:
            return y, x
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
                    

                    # Calculate the loss for each sample in the batch
                    losses = criterion(outputs, labels)
                    
                    # Apply the weighting for zero labels

                    
                    # Compute the mean loss
                    loss = torch.mean(losses)

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

    def evaluate_model(self, dataloader, criterion, device, save_results=False, results_dir=None, results_filename=None, zero_masking = False, zero_masking_model = None):
        # Calcualte the loss on the provided dataloader and save results if specified to H5 file including the input, output, and target
        self.eval()
        running_loss = 0.0
        results = {}

        if zero_masking and zero_masking_model is None:
            raise ValueError("zero_masking_model must be provided if zero_masking is True")

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
                if zero_masking and zero_masking_model is not None:
                    probs, zero_mask  = zero_masking_model.predict(inputs)
                    zero_mask = zero_mask.to(device)
                    # zero mask either 0 or 1
                    # change size of zero mask to match the size of the output dimensions so can broadcast in multiply
                    # print(zero_mask.shape)
                    zero_mask = torch.unsqueeze(zero_mask,2)
                    # print(zero_mask.shape)
                    zero_mask = zero_mask.to(device, torch.float32)

                    outputs = outputs * zero_mask


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
    def fuse_model(self):
        for name, module in [('encoder', self.encoder), ('decoder', self.decoder)]:
            for idx in range(len(module) - 1):
                if isinstance(module[idx], nn.Conv2d) and isinstance(module[idx + 1], nn.ReLU):
                    torch.ao.quantization.fuse_modules(module, [str(idx), str(idx + 1)], inplace=True)
                # elif isinstance(module[idx], nn.ConvTranspose2d) and isinstance(module[idx + 1], nn.ReLU):
                #     torch.ao.quantization.fuse_modules(module, [str(idx), str(idx + 1)], inplace=True)

    def quantize_static(self, calibration_dataloader):
        self.eval()
        self.fuse_model()
        self.qconfig = get_default_qconfig('fbgemm')

        prepared_model = prepare(self)

        with torch.no_grad():
            for batch in calibration_dataloader:
                inputs, _ = batch
                inputs = torch.unsqueeze(inputs, 1)
                inputs = inputs.to('cpu', dtype=torch.float32)
                prepared_model(inputs)

        quantized_model = convert(prepared_model)
        return quantized_model

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