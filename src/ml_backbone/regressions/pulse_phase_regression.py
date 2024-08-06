from regression_util import *


class RegressionModel(nn.Module):
    def __init__(self, 
                 fc_layers: List[List[Any]], 
                 conv_layers: List[List[Any]] = None, 
                 lstm_config: dict = None, 
                 dtype=torch.float32, 
                 use_dropout=False, 
                 dropout_rate=0.5):
        super(RegressionModel, self).__init__()
        self.dtype = dtype
        self.has_conv_layers = conv_layers is not None
        self.has_lstm_layers = lstm_config is not None

        if conv_layers is not None:
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

        if lstm_config is not None:
            # Create LSTM layer based on the provided configuration
            self.lstm_layers = nn.LSTM(input_size=lstm_config['input_size'], 
                                       hidden_size=lstm_config['hidden_size'], 
                                       num_layers=lstm_config['num_layers'], 
                                       bidirectional=lstm_config['bidirectional'], 
                                       batch_first=True)

            # Cast LSTM layers weights to specified dtype
            for param in self.lstm_layers.parameters():
                param.data = param.data.to(self.dtype)
        
        fc_modules = []
        for layer, activation in fc_layers:
            fc_modules.append(layer)
            if activation is not None:
                fc_modules.append(activation)
            if use_dropout:
                fc_modules.append(nn.Dropout(p=dropout_rate))
        
        self.fc_layers = nn.Sequential(*fc_modules)

        # Cast fc layers weights to specified dtype
        for param in self.fc_layers.parameters():
            param.data = param.data.to(self.dtype)

    def forward(self, x):
        if self.has_conv_layers:
            print(x.shape)
            x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
            x = self.conv_layers(x)
            if not self.has_lstm_layers:
                x = x.view(x.size(0), -1)
                
        
        if self.has_lstm_layers:
            x, (hn, cn) = self.lstm_layers(x)
            x = x[:, -1, :]  # Take the last output of the LSTM
        
        x = self.fc_layers(x)
        return x

        # # Initialize weights
        # for layer in self.hidden_layers:
        #     if isinstance(layer, nn.Linear):
        #         nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')

        # # Initialize biases to zeros
        # for layer in self.hidden_layers:
        #     if isinstance(layer, nn.Linear):
        #         nn.init.constant_(layer.bias, 0)
        # nn.init.constant_(self.output_layer.bias, 0)

    # def forward(self, x):
    #     if self.has_conv_layers:
    #         x = self.conv_layers(x)
    #         x = x.view(x.size(0), -1)

    #     x = self.fc_layers(x)

    #     return x

    def train_model(self, train_dataloader, val_dataloader, criterion, optimizer, scheduler, model_save_dir, identifier, device, 
                    checkpoints_enabled=True, resume_from_checkpoint=False, max_epochs=100, denoising=False,
                    denoise_model=None, zero_mask_model=None, lstm_pretrained_model=None, parallel=True, single_pulse=False):
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_epoch = 0
        start_epoch = 0
        if denoising and denoise_model is None and zero_mask_model is None:
            raise ValueError("Denoising is enabled but no denoising model is provided")
        if parallel:
            self = nn.DataParallel(self)
            if denoising and denoise_model is not None and zero_mask_model is not None:
                denoise_model = nn.DataParallel(denoise_model)
                zero_mask_model = nn.DataParallel(zero_mask_model)
                denoise_model.to(device)
                zero_mask_model.to(device)
        self.to(device)
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
        name = f"{model_save_dir}/{identifier}" + "_run_time_info.txt"
        with open(name, "a") as f:
            f.write(f"Training resumed at {datetime.datetime.now()} from epoch {start_epoch}\n" if start_epoch > 0 else f"Training started at {datetime.datetime.now()}\n")

            for epoch in range(start_epoch, max_epochs):
                self.train()  # Set the model to training mode
                running_train_loss = 0.0

                for batch in train_dataloader:
                    optimizer.zero_grad()  # Zero the parameter gradients

                    inputs, labels, phases = batch
                    inputs, labels, phases = inputs.to(device, self.dtype), labels.to(device, self.dtype), phases.to(device, self.dtype)
                    # print(labels)
                    if denoising and denoise_model is not None and zero_mask_model is not None:
                       
                        denoise_model.eval()
                        zero_mask_model.eval()
                        inputs = torch.unsqueeze(inputs, 1)
                        inputs = inputs.to(device, torch.float32)
                        # labels = labels[0]
                        outputs = denoise_model(inputs)
                        outputs = outputs.squeeze()
                        outputs = outputs.to(device)
                        if parallel:
                            probs, zero_mask  = zero_mask_model.module.predict(inputs)
                        else:
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
                    if lstm_pretrained_model is not None:
                        lstm_pretrained_model.eval()
                        inputs = lstm_pretrained_model(inputs)
                    outputs = self(inputs).to(device)
                    # print("outputs")
                    # print(outputs)
                    # print(outputs)
                    if single_pulse:
                        phases_differences= phases/(2*np.pi)
                        # print(phases)
                    else:   
                        phases_differences = (torch.abs(phases[:, 0] - phases[:, 1]))/(2*np.pi)
                    # print(phases_differences)
                    # print(outputs)
                    # loss = ((torch.cos(outputs*2*np.pi)-torch.cos(phases_differences*2*np.pi))**2 + (torch.sin(outputs*2*np.pi)-torch.sin(phases_differences*2*np.pi))**2).mean()
                    loss = criterion(outputs, phases)
                    loss.backward()
                    optimizer.step()

                    running_train_loss += loss.item()

                # train_loss = running_train_loss / (len(train_dataloader) + (len(second_train_dataloader) if second_train_dataloader else 0))
                train_loss = running_train_loss / (len(train_dataloader))

                train_losses.append(train_loss)
               

                # Validation loop
                self.eval()  # Set the model to evaluation mode
                running_val_loss = 0.0

                with torch.no_grad():
                    for batch in val_dataloader:
                        optimizer.zero_grad()  # Zero the parameter gradients

                        inputs, labels, phases = batch
                        inputs, labels, phases = inputs.to(device, self.dtype), labels.to(device,self.dtype), phases.to(device, self.dtype)
                        # print(labels)
                        if denoising and denoise_model is not None and zero_mask_model is not None:
                        
                            denoise_model.eval()
                            zero_mask_model.eval()
                            inputs = torch.unsqueeze(inputs, 1)
                            inputs = inputs.to(device, torch.float32)
                            # labels = labels[0]
                            outputs = denoise_model(inputs)
                            outputs = outputs.squeeze()
                            outputs = outputs.to(device)
                            if parallel:
                                probs, zero_mask  = zero_mask_model.module.predict(inputs)
                            else:
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
                        if lstm_pretrained_model is not None:
                            lstm_pretrained_model.eval()
                            inputs = lstm_pretrained_model(inputs)
                        outputs = self(inputs).to(device)
                        # print(outputs)
                        if single_pulse:
                            phases_differences= phases/(2*np.pi)
                            #    print(phases)
                        else:   
                            phases_differences = (torch.abs(phases[:, 0] - phases[:, 1]))/(2*np.pi)
                        phases_differences = phases_differences.to(device, self.dtype)
                        # loss = ((torch.cos(outputs*2*np.pi)-torch.cos(phases_differences*2*np.pi))**2 + (torch.sin(outputs*2*np.pi)-torch.sin(phases_differences*2*np.pi))**2).mean()
                        loss = criterion(outputs, phases)
                        running_val_loss += loss.item()
            
                val_loss = running_val_loss / len(val_dataloader)
                # val_loss = running_val_loss / (len(val_dataloader) + (len(second_val_dataloader) if second_val_dataloader else 0))

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



    def train_model_fromDenoise(self, train_dataloader, val_dataloader, criterion, optimizer, scheduler, model_save_dir, identifier, device, 
                    checkpoints_enabled=True, resume_from_checkpoint=False, max_epochs=100,
                    denoise_model=None, parallel=True):
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_epoch = 0
        start_epoch = 0
        
        if parallel:
            self = nn.DataParallel(self)
            denoise_model = nn.DataParallel(denoise_model)
            denoise_model.to(device)
        self.to(device)
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
        name = f"{model_save_dir}/{identifier}" + "_run_time_info.txt"
        with open(name, "a") as f:
            f.write(f"Training resumed at {datetime.datetime.now()} from epoch {start_epoch}\n" if start_epoch > 0 else f"Training started at {datetime.datetime.now()}\n")

            for epoch in range(start_epoch, max_epochs):
                self.train()  # Set the model to training mode
                running_train_loss = 0.0

                for batch in train_dataloader:
                    optimizer.zero_grad()  # Zero the parameter gradients

                    inputs, labels, phases = batch
                    inputs, labels, phases = inputs.to(device), labels.to(device), phases.to(device)
                    # print(labels)
                       
                    denoise_model.eval()
                    inputs = torch.unsqueeze(inputs, 1)
                    inputs = inputs.to(device, torch.float32)
                    outputs, decoded = denoise_model(inputs)
                    # outputs = outputs.view(outputs.size(0), -1)
                    
                        
                    
                    inputs = outputs.to(device, torch.float32)
                
                    outputs = self(inputs).to(device)
                    # print(outputs)
                    phases_differences = torch.abs(phases[:, 0] - phases[:, 1])
                    loss = ((torch.cos(outputs*2*np.pi)-torch.cos(phases_differences*2*np.pi))**2 + (torch.sin(outputs*2*np.pi)-torch.sin(phases_differences*2*np.pi))**2).mean()
                    # loss = criterion(outputs, phases)
                    loss.backward()
                    optimizer.step()

                    running_train_loss += loss.item()

                # train_loss = running_train_loss / (len(train_dataloader) + (len(second_train_dataloader) if second_train_dataloader else 0))
                train_loss = running_train_loss / (len(train_dataloader))

                train_losses.append(train_loss)
               

                # Validation loop
                self.eval()  # Set the model to evaluation mode
                running_val_loss = 0.0

                with torch.no_grad():
                    for batch in val_dataloader:
                        optimizer.zero_grad()  # Zero the parameter gradients

                        inputs, labels, phases = batch
                        inputs, labels, phases = inputs.to(device), labels.to(device), phases.to(device)
                        # print(labels)
                        
                        denoise_model.eval()
                        inputs = torch.unsqueeze(inputs, 1)
                        inputs = inputs.to(device, torch.float32)
                        outputs, decoded = denoise_model(inputs)
                        # outputs = outputs.view(outputs.size(0), -1)
                        
                            
                        inputs = outputs.to(device, torch.float32)

                        
                        outputs = self(inputs).to(device)
                        # print(outputs)
                        phases_differences = torch.abs(phases[:, 0] - phases[:, 1])
                        loss = ((torch.cos(outputs*2*np.pi)-torch.cos(phases_differences*2*np.pi))**2 + (torch.sin(outputs*2*np.pi)-torch.sin(phases_differences*2*np.pi))**2).mean()
                        running_val_loss += loss.item()
            
                val_loss = running_val_loss / len(val_dataloader)
                # val_loss = running_val_loss / (len(val_dataloader) + (len(second_val_dataloader) if second_val_dataloader else 0))

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



