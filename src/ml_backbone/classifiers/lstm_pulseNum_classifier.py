from classifiers_util import *

class CustomLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_lstm_layers, num_classes, bidirectional=False, fc_layers=None, dropout_p=0.5, lstm_dropout=0.2, layer_norm=False, ignore_output_layer=False):
        super(CustomLSTMClassifier, self).__init__()
        self.ignore_output_layer = ignore_output_layer

        # LSTM layers with optional bidirectionality and dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size * self.num_directions

        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_lstm_layers, 
            batch_first=True, 
            bidirectional=bidirectional, 
            dropout=lstm_dropout
        )

        # Optional layer normalization
        self.layer_norm = nn.LayerNorm(self.hidden_size) if layer_norm else None

        # Dropout layer before fully connected layers
        self.dropout = nn.Dropout(p=dropout_p)

        # Fully connected layers
        self.fc_layers, output_layer_input_size = self._build_fc_layers(fc_layers, hidden_size, layer_norm)
        
        # Output layer for classification
        self.output_layer = nn.Linear(output_layer_input_size, num_classes)

    def forward(self, x):
        # LSTM forward pass
        out, _ = self.lstm(x)

        # Apply layer normalization if defined
        if self.layer_norm:
            out = self.layer_norm(out)

        # Apply fully connected layers (if defined)
        if self.fc_layers is not None:
            out = self.fc_layers(out[:, -1, :])
            if self.ignore_output_layer:
                return out
        else:
            out = out[:, -1, :]
        
        # Final output layer for classification
        out = self.output_layer(out)

        return out

    def _build_fc_layers(self, fc_layers, hidden_size, layer_norm):
        if fc_layers is None:
            return None, self.hidden_size

        layers = []
        in_features = self.hidden_size
        for fc_layer_size in fc_layers:
            if fc_layer_size == 0:
                return None, hidden_size

            if fc_layer_size <= 0:
                raise ValueError("Fully connected layer size must be greater than 0")

            layers.append(nn.Linear(in_features, fc_layer_size))
            layers.append(nn.ReLU())  # Apply ReLU activation
            layers.append(self.dropout)  # Apply dropout

            if layer_norm:
                layers.append(nn.LayerNorm(fc_layer_size))

            in_features = fc_layer_size

        return nn.Sequential(*layers), fc_layers[-1]  # Last fully connected layer's output size
    
    def train_model(self, train_dataloader, val_dataloader, criterion, optimizer, scheduler, model_save_dir, identifier, device, checkpoints_enabled=True, resume_from_checkpoint=False, max_epochs=10, denoising=False, denoise_model = None, zero_mask_model = None, parallel=True):
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_epoch = 0
        start_epoch = 0

        print("Made it here")
        if parallel:
            self = nn.DataParallel(self)
        print("Made it here 2")

        self.to(device)
        checkpoint_path = os.path.join(model_save_dir, f"{identifier}_checkpoint.pth")

        if denoising and denoise_model is None and zero_mask_model is None:
            raise ValueError("Denoising is enabled but no denoising model is provided")
        
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
        name = f"{model_save_dir}/{identifier}" + "_run_time_info.txt"
        with open(name, "a") as f:
            f.write(f"Training resumed at {datetime.datetime.now()} from epoch {start_epoch}\n" if start_epoch > 0 else f"Training started at {datetime.datetime.now()}\n")

            for epoch in range(start_epoch, max_epochs):
                self.train()  # Set the model to training mode
                running_train_loss = 0.0

                for batch in train_dataloader:
                    optimizer.zero_grad()  # Zero the parameter gradients

                    inputs, labels = batch
                    labels = labels.to(device) #indexing for access to the first element of the list

                    if denoising and denoise_model is not None and zero_mask_model is not None:
                        denoise_model.eval()
                        zero_mask_model.eval()
                        inputs = torch.unsqueeze(inputs, 1)
                        inputs = inputs.to(device, torch.float32)
                        # labels = labels[0]
                        outputs = denoise_model(inputs)
                        outputs = outputs.squeeze()
                        outputs = outputs.to(device)
                        probs, zero_mask  = zero_mask_model.predict(inputs)
                        zero_mask = zero_mask.to(device)
                        # zero mask either 0 or 1
                        # change size of zero mask to match the size of the output dimensions so can broadcast in multiply
                        zero_mask = torch.unsqueeze(zero_mask,2)
                        zero_mask = zero_mask.to(device, torch.float32)

                        outputs = outputs * zero_mask
                        inputs = outputs.to(device, torch.float32)

                    else: 
                        inputs = inputs.to(device, torch.float32)
                    

                    outputs = self(inputs).to(device)
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
                        labels = labels.to(device) #indexing for access to the first element of the list

                        if denoising and denoise_model is not None and zero_mask_model is not None:
                            denoise_model.eval()
                            zero_mask_model.eval()
                            inputs = torch.unsqueeze(inputs, 1)
                            inputs = inputs.to(device, torch.float32)
                            # labels = labels[0]
                            outputs = denoise_model(inputs)
                            outputs = outputs.squeeze()
                            outputs = outputs.to(device)
                            probs, zero_mask  = zero_mask_model.predict(inputs)
                            zero_mask = zero_mask.to(device)
                            # zero mask either 0 or 1
                            # change size of zero mask to match the size of the output dimensions so can broadcast in multiply
                            zero_mask = torch.unsqueeze(zero_mask,2)
                            zero_mask = zero_mask.to(device, torch.float32)

                            outputs = outputs * zero_mask
                            inputs = outputs.to(device, torch.float32)

                        else: 
                            inputs = inputs.to(device, torch.float32)
                            

                        outputs = self(inputs).to(device)
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

                ## Save checkpoint
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

    def evaluate_model(self, test_dataloader, identifier, model_dir, device, denoising=False, denoise_model = None, zero_mask_model = None):
        # Lists to store true and predicted values for pulses
        true_1_pred_1 = []
        true_1_pred_2 = []
        true_2_pred_1 = []
        true_2_pred_2 = []
        true_2_pred_3 = []
        true_3_pred_2 = []
        true_3_pred_3 = []
        true_3_pred_4p = []

        true_pulses = []
        predicted_pulses = []

        self.to(device)

        if denoising and denoise_model is None and zero_mask_model is None:
            raise ValueError("Denoising is enabled but no denoising model is provided")
        self.eval()  # Set the model to evaluation mode, ensures no dropout is applied
        # Iterate through the test data
        with torch.no_grad():
            for inputs, labels in test_dataloader:
                # Move data to the GPU
                inputs, labels = inputs.to(device), labels.to(device)

                if denoising and denoise_model is not None and zero_mask_model is not None:
                    denoise_model.eval()
                    zero_mask_model.eval()
                    inputs = torch.unsqueeze(inputs, 1)
                    inputs = inputs.to(device, torch.float32)
                    # labels = labels[0]
                    outputs = denoise_model(inputs)
                    outputs = outputs.squeeze()
                    outputs = outputs.to(device)
                    probs, zero_mask  = zero_mask_model.predict(inputs)
                    zero_mask = zero_mask.to(device)
                    # zero mask either 0 or 1
                    # change size of zero mask to match the size of the output dimensions so can broadcast in multiply
                    zero_mask = torch.unsqueeze(zero_mask,2)
                    zero_mask = zero_mask.to(device, torch.float32)

                    outputs = outputs * zero_mask
                    inputs = outputs.to(device, torch.float32)

                else: 
                    inputs = inputs.to(device, torch.float32)
                
                probs, preds = self.predict(inputs)
                preds = preds.to(device)

                true_pulse_single_label = np.argmax(labels.cpu().numpy(), axis=1)
                predicted_pulse_single_label = np.argmax(preds.cpu().numpy(), axis=1)

                true_pulses.extend(true_pulse_single_label)
                predicted_pulses.extend(predicted_pulse_single_label)

        num_classes_from_test = len(np.unique(true_pulses))
        # Calculate evaluation metrics as percentages
        accuracy = accuracy_score(true_pulses, predicted_pulses) * 100
        precision = precision_score(true_pulses, predicted_pulses, average='macro') * 100
        recall = recall_score(true_pulses, predicted_pulses, average='macro') * 100
        f1 = f1_score(true_pulses, predicted_pulses, average='macro') * 100

        # Confusion matrix
        cm = confusion_matrix(true_pulses, predicted_pulses)

        # Normalize the confusion matrix based on percentages
        row_sums = cm.sum(axis=1, keepdims=True)
        normalized_cm = cm / row_sums.astype(float) * 100

        # Create class labels based on the number of classes
        class_labels = [f'{i} Pulse(s)' for i in range(num_classes_from_test)]

        # Plot the normalized confusion matrix with class labels
        plt.figure(figsize=(8, 6))
        plt.imshow(normalized_cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))

        # Add class labels to the plot
        plt.title('Normalized Confusion Matrix (%)')
        plt.colorbar()
        tick_marks = np.arange(len(class_labels))
        plt.xticks(tick_marks, class_labels, rotation=45)
        plt.yticks(tick_marks, class_labels)

        for i in range(num_classes_from_test):
            for j in range(num_classes_from_test):
                text_label = f"{normalized_cm[i, j]:.2f}%"
                plt.text(j, i, text_label, horizontalalignment="center", color="black")
        # Add axis labels
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plot_path = os.path.join(model_dir, identifier + "_confusion_matrix.pdf")
        # Display the plot
        plt.savefig(plot_path)
        plt.close()

        # Print and display metrics
        # Redirect output to the same log file used during training
        run_summary_path = f"{model_dir}/{identifier}"+ "_run_summary.txt"
        with open(run_summary_path, "a") as file:
            file.write(f"Accuracy: {accuracy:.2f}%\n")
            file.write(f"Precision: {precision:.2f}%\n")
            file.write(f"Recall: {recall:.2f}%\n")
            file.write(f"F1 Score: {f1:.2f}%\n")
            for i in range(num_classes_from_test):
                true_positives = cm[i, i]  # The diagonal of the confusion matrix
                total_instances = np.sum(cm[i, :])  # Total instances in the true class
                accuracy_class = true_positives / total_instances * 100

                class_name = f"Class {i}"
                file.write(f"{class_name} - Accuracy: {accuracy_class:.2f}%\n")

        return accuracy, precision, recall, f1, normalized_cm, plot_path
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
    
def create_model_from_json(json_file_path, input_size, num_classes, ignore_output_layer = True, load_weights = True):
    with open(json_file_path, 'r') as f:
        data =  json.load(f)
    data = data['best_results_formatted_parameters']
    
    # Create CustomLSTMClassifier model
    model = CustomLSTMClassifier(
        input_size=input_size,
        hidden_size=data['hidden_size'],
        num_lstm_layers=data['num_lstm_layers'],
        num_classes= num_classes,
        bidirectional=data['bidirectional'],
        fc_layers=data['fc_layers'],
        dropout_p=data['dropout'],
        lstm_dropout=data['lstm_dropout'],
        layerNorm=data['layerNorm'],
        ignore_output_layer = ignore_output_layer
    )
    
    # Load model weights
    if load_weights:
        path = data['path_to_best_model_weights']
        model.load_state_dict(torch.load(data['path_to_best_model_weights']))
        return model, path, data
    else:
        return model, None, data    