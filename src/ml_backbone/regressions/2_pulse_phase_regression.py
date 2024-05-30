# Load all the necessary libraries
# import utils_1
from utils_1 import *
import utils_1

# Check if CUDA (GPU support) is available
print("CUDA is available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU is available!")
    device = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
    print("MPS is available. Using GPU.")
else:
    device = torch.device("cpu")
    print("MPS is not available. Using CPU.")

original_stdout = sys.stdout

def train_model(model, num_epochs, train_dataloader, val_dataloader, criterion, optimizer, scheduler, patience_earlyStop, filename, model_dir):
    # Lists to store training and validation losses for later plotting
    train_losses = []
    val_losses = []

    # Early stopping parameters
    early_stopping_counter = 0
    best_val_loss = float('inf')
    best_model = None


    name = filename + "_run_time_info.txt"
    name = os.path.join(model_dir, name)
    with open(name, "w") as file:
        sys.stdout = file

        # Training loop
        for epoch in range(num_epochs):
            model.train()  # Set the model to training mode
            running_train_loss = 0.0

            for batch in train_dataloader:
                optimizer.zero_grad()  # Zero the parameter gradients

                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_train_loss += loss.item()

            train_loss = running_train_loss / len(train_dataloader)
            train_losses.append(train_loss)

            # Validation loop
            model.eval()  # Set the model to evaluation mode
            running_val_loss = 0.0
            
            with torch.no_grad():
                for batch in val_dataloader:
                    inputs, labels = batch
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    running_val_loss += loss.item()

            val_loss = running_val_loss / len(val_dataloader)
            val_losses.append(val_loss)

            print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.10f}, Validation Loss: {val_loss:.10f}")
            
            # Learning rate scheduling
            scheduler.step(val_loss)

             # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stopping_counter = 0
                best_model = copy.deepcopy(model.state_dict())
                best_epoch = epoch
            else:
                early_stopping_counter += 1

            if early_stopping_counter >= patience_earlyStop:
                print("Early stopping: No improvement in validation loss for {} epochs.".format(patience_earlyStop))
                break  # Stop training

            # Load the best model
        if best_model is not None:
            model.load_state_dict(best_model)
            # Save the best model with a specified name and path in the model_dir
            name = filename + "_best_model.pth"
            best_model_path = os.path.join(model_dir, name)
            torch.save(model.state_dict(), best_model_path)

        # Save the output to the specified file
        name = filename + "_run_summary.txt"
        run_summary_path = os.path.join(model_dir, name)
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
        losses_path = os.path.join(model_dir, filename + "_losses.pdf")
        plt.savefig(losses_path)
        plt.close()

    sys.stdout = original_stdout 
    return best_epoch + 1, best_val_loss, best_model_path, run_summary_path, losses_path, np.array(train_losses), np.array(val_losses), train_losses[best_epoch], val_losses[best_epoch]


# def train_model_encoder_regression(model_stationary, model, num_epochs, train_dataloader, val_dataloader, criterion, optimizer, scheduler, patience_earlyStop, filename, model_dir, model_params=None, optimizer_type=None):
#     # Lists to store training and validation losses for later plotting
#     train_losses = []
#     val_losses = []

#     # Early stopping parameters
#     early_stopping_counter = 0
#     best_val_loss = float('inf')
#     best_model = None


#     name = filename + "_run_time_info.txt"
#     name = os.path.join(model_dir, name)
#     with open(name, "w") as file:
#         sys.stdout = file
#         print("Model Parameters:")
#         print(model_params)
#         print("Optimizer Type:")
#         print(optimizer_type)
#         print("Model Architecture:")
#         print(model)
#         print("Stationary Model Architecture:")
#         print(model_stationary)
#         # Training loop
#         for epoch in range(num_epochs):
#             model.train()  # Set the model to training mode
#             running_train_loss = 0.0

#             for batch in train_dataloader:
#                 optimizer.zero_grad()  # Zero the parameter gradients

#                 inputs, labels = batch
#                 inputs, labels = inputs.to(device), labels.to(device)

#                 # Pass inputs through the stationary model
#                 with torch.no_grad():
#                     stationary_outputs = model_stationary(inputs)
                
#                 batch_size, num_channels, height, width = stationary_outputs.size()
               
#                 flattened_outputs = stationary_outputs.view(batch_size, -1)


#                 # Pass stationary outputs through the second model
#                 outputs = model(flattened_outputs)
#                 a = (outputs[:, 0]) * torch.tensor(np.pi * 2, dtype=torch.float32, device=outputs.device)
#                 label_ph_diff = (torch.abs((labels[:, 0]*np.pi*2  + np.pi) % (2 * np.pi) - np.pi) *torch.sign(labels[:, 0]))
#                 b = label_ph_diff
      
#                 # Calculate the quantities
#                 phase_loss = (torch.cos(a) - torch.cos(b))**2 + (torch.sin(a) - torch.sin(b))**2
#                 phase_loss = torch.mean(phase_loss)
                
#                 # energy_loss = criterion(outputs[:,1:], labels[:,2:])
#                 # energy_loss = ut.torch.mean(energy_loss)

                
#                 loss = phase_loss# + energy_loss
#                 loss.backward()
#                 # loss = criterion(outputs, labels)
#                 # loss.backward()
#                 optimizer.step()

#                 running_train_loss += loss.item()

#             train_loss = running_train_loss / len(train_dataloader)
#             train_losses.append(train_loss)

#             # Validation loop
#             model.eval()  # Set the model to evaluation mode
#             running_val_loss = 0.0
            
#             with torch.no_grad():
#                 for batch in val_dataloader:
#                     inputs, labels = batch
#                     inputs, labels = inputs.to(device), labels.to(device)
#                     # Pass inputs through the stationary model
#                     stationary_outputs = model_stationary(inputs)
#                     batch_size, num_channels, height, width = stationary_outputs.size()
#                     flattened_outputs = stationary_outputs.view(batch_size, -1)
                
#                     # Pass stationary outputs through the second model
#                     outputs = model(flattened_outputs)
#                     a = (outputs[:, 0]) * torch.tensor(np.pi * 2, dtype=torch.float32, device=outputs.device)
#                     label_ph_diff = (torch.abs((labels[:, 0]*np.pi*2  + np.pi) % (2 * np.pi) - np.pi) *torch.sign(labels[:, 0]))
#                     b = label_ph_diff
        
#                     # Calculate the quantities
#                     phase_loss = (torch.cos(a) - torch.cos(b))**2 + (torch.sin(a) - torch.sin(b))**2
#                     phase_loss = torch.mean(phase_loss)
                    
#                     # energy_loss = criterion(outputs[:,1:], labels[:,2:])
#                     # energy_loss = ut.torch.mean(energy_loss)

                    
#                     loss = phase_loss# + energy_loss
#                     # loss = criterion(outputs, labels)
#                     running_val_loss += loss.item()

#             val_loss = running_val_loss / len(val_dataloader)
#             val_losses.append(val_loss)

#             # print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.10f}, Validation Loss: {val_loss:.10f}")
#                     # Print the learning rate
#             for param_group in optimizer.param_groups:
#                 current_lr = param_group['lr']
#                 print(f"Epoch [{epoch+1}/{num_epochs}] - Learning Rate: {current_lr}, Train Loss: {train_loss:.10f}, Validation Loss: {val_loss:.10f}")

#             # Learning rate scheduling
#             scheduler.step(val_loss)

#                 # Early stopping
#             if val_loss < best_val_loss:
#                 best_val_loss = val_loss
#                 early_stopping_counter = 0
#                 best_model = copy.deepcopy(model.state_dict())
#                 best_epoch = epoch
#             else:
#                 early_stopping_counter += 1

#             if early_stopping_counter >= patience_earlyStop:
#                 print("Early stopping: No improvement in validation loss for {} epochs.".format(patience_earlyStop))
#                 break  # Stop training

#             # Load the best model
#         if best_model is not None:
#             model.load_state_dict(best_model)
#             # Save the best model with a specified name and path in the model_dir
#             name = filename + "_best_model.pth"
#             best_model_path = os.path.join(model_dir, name)
#             torch.save(model.state_dict(), best_model_path)

#         # Save the output to the specified file
#         name = filename + "_run_summary.txt"
#         run_summary_path = os.path.join(model_dir, name)
#         with open(run_summary_path, "w") as file:
#             file.write("Number of Epochs for Best Model: {}\n".format(best_epoch + 1))
#             file.write("Final Training Loss: {:.10f}\n".format(train_losses[-1]))
#             file.write("Final Validation Loss: {:.10f}\n".format(val_losses[-1]))

#         # Plot the training and validation losses
#         plt.figure()
#         plt.plot(train_losses, label='Train Loss')
#         plt.plot(val_losses, label='Validation Loss')
#         plt.scatter(best_epoch, val_losses[best_epoch], marker='*', color='red', label='Best Epoch')
#         plt.xlabel('Epoch')
#         plt.ylabel('Loss')
#         plt.title('Training and Validation Loss')
#         plt.legend()
#         losses_path = os.path.join(model_dir, filename + "_losses.pdf")
#         plt.savefig(losses_path)
#         plt.close()

#     sys.stdout = original_stdout 
#     return best_epoch + 1, best_val_loss, best_model_path, run_summary_path, losses_path, np.array(train_losses), np.array(val_losses), train_losses[best_epoch], val_losses[best_epoch]
def train_model_encoder_regression(model_stationary, model, num_epochs, train_dataloader, val_dataloader, criterion, optimizer, scheduler, patience_earlyStop, filename, model_dir, model_params=None, optimizer_type=None):
    # Lists to store training and validation losses for later plotting
    train_losses = []
    val_losses = []

    # Early stopping parameters
    early_stopping_counter = 0
    best_val_loss = float('inf')
    best_model = None
    reinitialize_weights = True

    name = filename + "_run_time_info.txt"
    name = os.path.join(model_dir, name)
    with open(name, "w") as file:
        sys.stdout = file
        print("Model Parameters:")
        print(model_params)
        print("Optimizer Type:")
        print(optimizer_type)
        print("Model Architecture:")
        print(model)
        print("Stationary Model Architecture:")
        print(model_stationary)
        # Training loop
        epoch = 0
        # for epoch in range(num_epochs):
        while epoch < num_epochs:
            model.train()  # Set the model to training mode
            running_train_loss = 0.0

            for batch in train_dataloader:
                optimizer.zero_grad()  # Zero the parameter gradients

                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)

                # Pass inputs through the stationary model
                with torch.no_grad():
                    stationary_outputs = model_stationary(inputs)
                
                batch_size, num_channels, height, width = stationary_outputs.size()
               
                flattened_outputs = stationary_outputs.view(batch_size, -1)


                # Pass stationary outputs through the second model
                outputs = model(flattened_outputs)
                a = (outputs[:, 0]) * torch.tensor(np.pi * 2, dtype=torch.float32, device=outputs.device)
                label_ph_diff = (torch.abs((labels[:, 0]*np.pi*2  + np.pi) % (2 * np.pi) - np.pi) *torch.sign(labels[:, 0]))
                b = label_ph_diff
      
                # Calculate the quantities
                phase_loss = (torch.cos(a) - torch.cos(b))**2 + (torch.sin(a) - torch.sin(b))**2
                phase_loss = torch.mean(phase_loss)
                
                # energy_loss = criterion(outputs[:,1:], labels[:,2:])
                # energy_loss = ut.torch.mean(energy_loss)

                
                loss = phase_loss# + energy_loss
                loss.backward()
                # loss = criterion(outputs, labels)
                # loss.backward()
                optimizer.step()

                running_train_loss += loss.item()

            train_loss = running_train_loss / len(train_dataloader)
            train_losses.append(train_loss)

            # Validation loop
            model.eval()  # Set the model to evaluation mode
            running_val_loss = 0.0
            
            with torch.no_grad():
                for batch in val_dataloader:
                    inputs, labels = batch
                    inputs, labels = inputs.to(device), labels.to(device)
                    # Pass inputs through the stationary model
                    stationary_outputs = model_stationary(inputs)
                    batch_size, num_channels, height, width = stationary_outputs.size()
                    flattened_outputs = stationary_outputs.view(batch_size, -1)
                
                    # Pass stationary outputs through the second model
                    outputs = model(flattened_outputs)
                    a = (outputs[:, 0]) * torch.tensor(np.pi * 2, dtype=torch.float32, device=outputs.device)
                    label_ph_diff = (torch.abs((labels[:, 0]*np.pi*2  + np.pi) % (2 * np.pi) - np.pi) *torch.sign(labels[:, 0]))
                    b = label_ph_diff
        
                    # Calculate the quantities
                    phase_loss = (torch.cos(a) - torch.cos(b))**2 + (torch.sin(a) - torch.sin(b))**2
                    phase_loss = torch.mean(phase_loss)
                    
                    # energy_loss = criterion(outputs[:,1:], labels[:,2:])
                    # energy_loss = ut.torch.mean(energy_loss)

                    
                    loss = phase_loss# + energy_loss
                    # loss = criterion(outputs, labels)
                    running_val_loss += loss.item()

            val_loss = running_val_loss / len(val_dataloader)
            val_losses.append(val_loss)

            # print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.10f}, Validation Loss: {val_loss:.10f}")
                    # Print the learning rate
            for param_group in optimizer.param_groups:
                current_lr = param_group['lr']
                print(f"Epoch [{epoch+1}/{num_epochs}] - Learning Rate: {current_lr}, Train Loss: {train_loss:.10f}, Validation Loss: {val_loss:.10f}")

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stopping_counter = 0
                best_model = copy.deepcopy(model.state_dict())
                best_epoch = epoch
            else:
                early_stopping_counter += 1

            if early_stopping_counter >= patience_earlyStop:
                print("Early stopping: No improvement in validation loss for {} epochs.".format(patience_earlyStop))
                break  # Stop training

            # Reinitialize weights if no improvement in first five epochs
            if early_stopping_counter == 5 and epoch <= 6 and reinitialize_weights:
                print("Reinitializing weights...")
                model.apply(weights_init)
                reinitialize_weights = False
                epoch = 0

            epoch += 1
            # Load the best model
        if best_model is not None:
            model.load_state_dict(best_model)
            # Save the best model with a specified name and path in the model_dir
            name = filename + "_best_model.pth"
            best_model_path = os.path.join(model_dir, name)
            torch.save(model.state_dict(), best_model_path)

        # Save the output to the specified file
        name = filename + "_run_summary.txt"
        run_summary_path = os.path.join(model_dir, name)
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
        losses_path = os.path.join(model_dir, filename + "_losses.pdf")
        plt.savefig(losses_path)
        plt.close()

    sys.stdout = original_stdout 
    return best_epoch + 1, best_val_loss, best_model_path, run_summary_path, losses_path, np.array(train_losses), np.array(val_losses), train_losses[best_epoch], val_losses[best_epoch]

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
# Get Phase Difference and ypdf

# CNN Regression model

# LSTM Regression model 


# Autoencoder model
# Define the autoencoder architecture
# class Autoencoder(nn.Module):
#     def __init__(self):
#         super(Autoencoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=3, padding=2),
#             nn.ReLU(),
#             nn.Conv2d(16, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#         )
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(16, 1, kernel_size=3, padding=2),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x
    
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*4*32, 256),  # Compressed representation
            nn.ReLU()
            nn.Linear(256, 64)
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            # nn.Linear(256, 64*4*32),
            # nn.ReLU(),
            # nn.Unflatten(1, (64, 4, 32)),  # Reshape to 4x32x64
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class RegressionModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, activations, use_batch_norm=False, batch_norm_momentum=0.1, use_dropout=False, dropout_rate=0.5):
        super(RegressionModel, self).__init__()
        self.hidden_layers = nn.ModuleList()
        self.use_batch_norm = use_batch_norm
        self.batch_norm_momentum = batch_norm_momentum
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate

        prev_size = input_size
        for hidden_size, activation in zip(hidden_sizes, activations):
            self.hidden_layers.append(nn.Linear(prev_size, hidden_size))
            if self.use_batch_norm:
                self.hidden_layers.append(nn.BatchNorm1d(hidden_size, momentum=self.batch_norm_momentum))
            if activation == 'relu':
                self.hidden_layers.append(nn.ReLU())
            elif activation == 'sigmoid':
                self.hidden_layers.append(nn.Sigmoid())
            elif activation == 'tanh':
                self.hidden_layers.append(nn.Tanh())
            if self.use_dropout:
                self.hidden_layers.append(nn.Dropout(self.dropout_rate))
            prev_size = hidden_size
        self.output_layer = nn.Linear(prev_size, 1)

        # Initialize weights
        for layer in self.hidden_layers:
            if isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')

        # Initialize biases to zeros
        for layer in self.hidden_layers:
            if isinstance(layer, nn.Linear):
                init.constant_(layer.bias, 0)
        init.constant_(self.output_layer.bias, 0)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x
# Custom Functions
def calculate_scaler(file_paths, scaler_save_path, scaler_name):
    """
    Calculate and save a MinMaxScaler based on the data in the specified files.

    Args:
    - file_paths (list): List of file paths containing data for scaling.
    - scaler_save_path (str): Path where the scaler should be saved.

    Returns:
    - scaler (MinMaxScaler): The trained MinMaxScaler.
    """
    scaler = MinMaxScaler(feature_range=(-1, 1))

    for file_path in file_paths:
        with h5py.File(file_path, 'r') as h5f:
            for image_key in h5f.keys():
                image = h5f[image_key]["Ximg"][:]
                scaler.partial_fit(image)

    # Save the scaler to a file
    full_scaler_save_path = os.path.join(scaler_save_path, scaler_name)
    joblib.dump(scaler, full_scaler_save_path)

    return scaler

def load_and_preprocess_data(file_paths, energy_elements=512, scaler=None, scaler_save_path=None, scaler_only=False, scaler_additional_identifier=None):
    images = []
    images_minMaxScaled = []
    energies = []
    energies_normalized = []
    phases = []
    phases_normalized = []
    ypdfs = []
    ypdfs_minMaxScaled = []
    npulses = []
    nphases = []
    nenergies = []

    if scaler is None:
        min_max_scaler = MinMaxScaler(feature_range=(0, 1))
        min_max_scaler_ypdf = MinMaxScaler(feature_range=(0, 1))
        
        for file_path in file_paths:
            with h5py.File(file_path, 'r') as h5f:
                for image_key in h5f.keys():
                    min_max_scaler.partial_fit(h5f[image_key]["Ximg"][:])
                    min_max_scaler_ypdf.partial_fit(h5f[image_key]["Ypdf"][:])
        if scaler_save_path is not None:
            if scaler_additional_identifier is not None:
                orig_name = scaler_additional_identifier+"_min_max_scaler.joblib"
                orig_ypdf_name = scaler_additional_identifier+"_min_max_ypdf_scaler.joblib"
            else:   
                orig_name = "min_max_scaler.joblib"
                orig_ypdf_name = "min_max_ypdf_scaler.joblib"
            min_max_file = os.path.join(scaler_save_path, orig_name)
            min_max_ypdf_file = os.path.join(scaler_save_path, orig_ypdf_name)
            joblib.dump(min_max_scaler, min_max_file)
            joblib.dump(min_max_scaler_ypdf, min_max_ypdf_file)
    else:
        min_max_scaler = scaler["min_max_scaler"]
        min_max_scaler_ypdf = scaler["min_max_scaler_ypdf"]
    if scaler_only:
        return min_max_scaler, min_max_scaler_ypdf
    for file_path in file_paths:
        with h5py.File(file_path, 'r') as h5f:
            for image_key in h5f.keys():
                npulse = h5f[image_key].attrs["sasecenters"].shape[0]
                energy = h5f[image_key].attrs["sasecenters"]
                phase = h5f[image_key].attrs["sasephases"]
                image = h5f[image_key]["Ximg"][:]
                ypdf = h5f[image_key]["Ypdf"][:]
                min_max_scaled_image = min_max_scaler.transform(image)
                min_max_scaled_ypdf = min_max_scaler_ypdf.transform(ypdf)
                    
                images.append(image)
                energies.append(energy)
                energies_normalized.append(1/1+np.exp(-0.1*(energy-256))) # scale with sigmoid (average is 256, where min is around 234 and max is around 275)
                phases.append(phase)
                phases_normalized.append(phase/(2*np.pi))
                ypdfs.append(ypdf)
                images_minMaxScaled.append(min_max_scaled_image)
                ypdfs_minMaxScaled.append(min_max_scaled_ypdf)
                    
    images = np.asarray(images)
    images_minMaxScaled = np.asarray(images_minMaxScaled)
    ypdfs = np.asarray(ypdfs)
    ypdfs_minMaxScaled = np.asarray(ypdfs_minMaxScaled)
    phases = np.asarray(phases)
    phases_normalized = np.asarray(phases_normalized)
    energies = np.asarray(energies)
    energies_normalized = np.asarray(energies_normalized)
    
    return images, images_minMaxScaled, ypdfs, ypdfs_minMaxScaled, npulses, phases, phases_normalized, energies,energies_normalized, min_max_scaler, min_max_scaler_ypdf


def write_args_to_file(args, output_dir, filename="args_summary.txt"):
    """
    Write the command line arguments to a summary file in the output directory.

    Args:
    - args (argparse.Namespace): The parsed command line arguments.
    - output_dir (str): The directory where the summary file should be saved.
    - filename (str): The name of the summary file (default: "args_summary.txt").
    """
    # Create a summary file in the output directory
    summary_filepath = os.path.join(output_dir, filename)

    # Get the current timestamp
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    # Write the argument summary to the summary file, including the timestamp
    with open(summary_filepath, 'w') as summary_file:
        summary_file.write("Command Line Arguments:\n")
        summary_file.write(f"Timestamp: {timestamp}\n")  # Add the timestamp
        for arg in vars(args):
            summary_file.write(f"{arg}: {getattr(args, arg)}\n")


def create_sequences_and_targets(images, pulse_counts, nphases, nenergies, threshold=3, using_noise_correction=False):
    """
    Create sequences and targets for a classification dataset based on image data and pulse counts.

    Args:
    - images (list): List of preprocessed images.
    - pulse_counts (list): List of pulse counts corresponding to the images.
    - threshold (int): Threshold for defining classes (default: 3).

    Returns:
    - dataset (TensorDataset): Dataset containing sequences (images) and targets (pulse counts).
    """
    sequences = []
    targets = []

    # Define the classes for classification based on the threshold
    if threshold == 3:
        classes = ["0", "1", "2", "3", "beyond"]
    elif threshold == 2:
        classes = ["0", "1", "2", "beyond"]
    elif threshold == 1:
        classes = ["0", "1", "beyond"]

    # Encode the classes using LabelEncoder
    label_encoder = LabelEncoder()
    label_encoder.fit(classes)

    for i in range(len(images)):
        # The last element in the sequence is the target label
        target = pulse_counts[i]

        # Map the pulse counts into classes
        if target <= threshold:
            target_class = label_encoder.transform([f"{target}"])[0]
        else:
            target_class = label_encoder.transform(["beyond"])[0]
        # print(target_class)
        # Pad or truncate nphases and nenergies to a maximum length of 6
        if target_class == 0:
            padded_nphases = np.zeros(6)
            padded_nenergies = np.zeros(6)
        elif len(nphases[i]) <= 6:
            padded_nphases = np.pad(nphases[i], (0, 6 - len(nphases[i])), mode='constant')
            padded_nenergies = np.pad(nenergies[i], (0, 6 - len(nenergies[i])), mode='constant')
        else:
            padded_nphases = np.pad(nphases[i][0:6], (0, 0), mode='constant')
            padded_nenergies = np.pad(nenergies[i][0:6], (0, 0), mode='constant')
        

        target = [target_class] + list(padded_nphases) + list(padded_nenergies)
  
        targets.append(target)

    # Convert the list of targets to a numpy array
    targets_array = np.array(targets, dtype=np.float32)

    # Convert the list of sequences and targets to numpy arrays
    images_array = np.array(images, dtype=np.float32)
    # print(targets)
    targets_array = np.array(targets, dtype=np.float32)


    # Create PyTorch tensors from the numpy arrays
    sequences = torch.from_numpy(images_array)
    targets = torch.from_numpy(targets_array)

    # Create a dataset using the tensors
    dataset = TensorDataset(sequences, targets)

    if using_noise_correction:
        denoise_images_array = np.expand_dims(images_array, axis=1)
        denoise_sequences = torch.from_numpy(denoise_images_array)
        denoise_dataset = TensorDataset(denoise_sequences, targets)
        return dataset, denoise_dataset

    return dataset

def create_sequences_and_targets_encoder_regression(images, phases):
    # Create PyTorch tensors from the numpy arrays
    images = np.array(images, dtype= np.float32)
    images = np.expand_dims(images, axis=1)
    phases = np.array(phases, dtype= np.float32)
    phases_difference = np.diff(phases, axis=1)
    normalized_phase_difference = phases_difference/(2*np.pi)
    sequences = torch.from_numpy(images)
    targets = torch.from_numpy(normalized_phase_difference)

    # Create a dataset using the tensors
    dataset =TensorDataset(sequences, targets)
    
    return dataset


def create_sequences_and_targets_autoencoder(images):

    # Create PyTorch tensors from the numpy arrays
    images = np.array(images, dtype= np.float32)
    images = np.expand_dims(images, axis=1)
    # ypdfs = np.array(ypdfs, dtype= ut.np.float32)
    # ypdfs = np.expand_dims(ypdfs, axis=1)
    sequences = torch.from_numpy(images)
    targets = torch.from_numpy(images) #mapping same to same

    # Create a dataset using the tensors
    dataset =TensorDataset(sequences, targets)
    
    return dataset
def main():
    # Load dataset and create dataloaders
    # Create output directory for analysis
    dir_name = "/sdf/data/lcls/ds/prj/prjs2e21/results/MRCO_ML_Output/"
    output_dir = os.path.join(dir_name, "2_Pulse_Regression_05-01-2024")

    # Check if the directory exists, if not, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    two_pulse_data = "/sdf/data/lcls/ds/prj/prjs2e21/results/2-Pulse_04232024"
    two_pulse_scaler_save_path = "/sdf/data/lcls/ds/prj/prjs2e21/results/2-Pulse_04232024/"
    two_pulse_scaler_path = "/sdf/data/lcls/ds/prj/prjs2e21/results/2-Pulse_04232024/min_max_scaler.joblib"
    two_pulse_ypdf_scaler_path = "/sdf/data/lcls/ds/prj/prjs2e21/results/2-Pulse_04232024/min_max_ypdf_scaler.joblib"

    data_file_paths = [os.path.join(two_pulse_data, file) for file in os.listdir(two_pulse_data) if file.endswith('.h5')]
    # data_file_paths = data_file_paths[0:2]
    calc_scaler = False
    # Check if scaler file exists
    if os.path.exists(two_pulse_scaler_path) and os.path.exists(two_pulse_ypdf_scaler_path) and not calc_scaler:
        # Load the scaler
        print("Loading scaler...")
        scaler = joblib.load(two_pulse_scaler_path)
        scaler_ypdf = joblib.load(two_pulse_ypdf_scaler_path)
        # scaler.clip = False  # For compatibility with older versions of joblib
    else:
        # Calculate the scaler and save it
        scaler, scaler_ypdf = load_and_preprocess_data(file_paths=data_file_paths, energy_elements=512, scaler=None, scaler_save_path=two_pulse_scaler_save_path, scaler_only=True, scaler_additional_identifier="scaler_2_pulse")

    scaler_combined = {"min_max_scaler": scaler, "min_max_scaler_ypdf": scaler_ypdf}
    # Load and Process Even Data
    images, images_minMaxScaled, ypdfs, ypdfs_minMaxScaled,npulses, phases, phases_normalized, energies,energies_normalized, min_max_scaler, min_max_scaler_ypdf = load_and_preprocess_data(file_paths=data_file_paths, energy_elements=512, scaler=scaler_combined)

    input_size_image = images[0].shape[1]
    dataset_autoencorder = create_sequences_and_targets_autoencoder(images=ypdfs_minMaxScaled)
    # Check that the split ratios add up to 1.0
    split_ratios = check_split_ratios([.8,.1,.1])
    # Calculate the sizes of train, validation, and test sets
    train_ratio, val_ratio, test_ratio = split_ratios
    total_size = len(dataset_autoencorder)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    print(f"Train Size: {train_size}, Val Size: {val_size}, Test Size: {test_size}")
    torch.manual_seed(42)
    train_autoencoder_dataset, val_autoencoder_dataset, test_autoencoder_dataset = random_split(dataset_autoencorder, [train_size, val_size, test_size])
    batch_size = 32
    autoencoder_train_dataloader = DataLoader(train_autoencoder_dataset, batch_size=batch_size, shuffle=True)
    autoencoder_val_dataloader = DataLoader(val_autoencoder_dataset, batch_size=batch_size, shuffle=False)
    autoencoder_test_dataloader = DataLoader(test_autoencoder_dataset, batch_size=batch_size, shuffle=False)

    dataset_encoder_regression = create_sequences_and_targets_encoder_regression(images=ypdfs_minMaxScaled, phases=phases)
    # Check that the split ratios add up to 1.0
    split_ratios = check_split_ratios([.8,.1,.1])
    # Calculate the sizes of train, validation, and test sets
    train_ratio, val_ratio, test_ratio = split_ratios
    total_size = len(dataset_encoder_regression)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    print(f"Train Size: {train_size}, Val Size: {val_size}, Test Size: {test_size}")
    torch.manual_seed(42)
    train_encoder_regression_dataset, val_encoder_regression_dataset, test_encoder_regression_dataset = random_split(dataset_encoder_regression, [train_size, val_size, test_size])
    batch_size = 32
    encoder_regression_train_dataloader = DataLoader(train_encoder_regression_dataset, batch_size=batch_size, shuffle=True)
    encoder_regression_val_dataloader = DataLoader(val_encoder_regression_dataset, batch_size=batch_size, shuffle=False)
    encoder_regression_test_dataloader = DataLoader(test_encoder_regression_dataset, batch_size=batch_size, shuffle=False)
    # Train Autoencoder model

    # Define the loss function and optimizer for autoencoder
    autoencoder = Autoencoder().to(device)
    criterion_autoencoder = nn.MSELoss()
    optimizer_autoencoder = optim.Adam(autoencoder.parameters(), lr=0.001)


    num_epochs = 200
    initial_lr = 0.001
    optimizer = 'Adam'
    if optimizer == 'Adam':
        optimizer = optim.Adam(autoencoder.parameters(), lr=initial_lr)
    elif optimizer == 'SGD':
        optimizer = optim.SGD(autoencoder.parameters(), lr=initial_lr)
    elif optimizer == 'RMSprop':
        optimizer = optim.RMSprop(autoencoder.parameters(), lr=initial_lr)
    elif optimizer == 'AdamW':
        optimizer = optim.AdamW(autoencoder.parameters(), lr=initial_lr)
    else:
        raise ValueError("Optimizer type must be one of 'Adam', 'SGD', 'RMSprop', or 'AdamW'")
    lr_reduction_factor = 0.5       
    patience_learningRate = 4
    patience_earlyStop = 13
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=lr_reduction_factor, patience=patience_learningRate, verbose=True)

    filename = 'autoencoder'
    # Check if we want to retrain and if weights exist
    retrain = False

    output_dir_auto = os.path.join(dir_name, "2_Pulse_Regression_04-30-2024")

    autoencoder_weights_path = os.path.join(output_dir_auto, "autoencoder_best_model.pth")
    if not retrain and os.path.exists(autoencoder_weights_path):
        print("Loading trained autoencoder model...")

        # Load the weights
        autoencoder.load_state_dict(torch.load(autoencoder_weights_path))
        # Set encoder to just the encoder part
        encoder = autoencoder.encoder
        print("******** Weights loaded, encoder set. ***********")
    else:
        print("Training autoencoder model...")
        best_epoch_readable, best_val_loss, best_model_path, run_summary_path, losses_path, train_loss_array, val_loss_array, best_epoch_train_loss, best_epoch_val_loss  = train_model(model=autoencoder, num_epochs=num_epochs, train_dataloader=autoencoder_train_dataloader,
                                                                                                        val_dataloader=autoencoder_val_dataloader, criterion=criterion_autoencoder, optimizer=optimizer,scheduler=scheduler, patience_earlyStop=patience_earlyStop,
                                                                                                        filename=filename, model_dir=output_dir)
        autoencoder_path =  os.path.join(output_dir, 'autoencoder_best_model.pth')
        autoencoder.load_state_dict(torch.load(autoencoder_path))   
        encoder = autoencoder.encoder
        print("******** Weights loaded, encoder set. ***********")



    print(encoder)
    # Assuming you have an example input tensor
    example_input = torch.randn(1, 1, 512, 16)  # Batch size of 1, 1 channel, 512x16 input size
    example_input = example_input.to(device)
    encoder.to(device)
    # Pass the input through the encoder
    encoder_output = encoder(example_input)

    # Get the flattened size from the encoder
    flattened_size = encoder_output.view(encoder_output.size(0), -1).size(1)
    print("Flattened size of encoder output:", flattened_size)

    # Initialize regression model and move to device
    # regression_model = RegressionModel(input_size=64 * 4 * 32, hidden_size=64).to(device)
    # regression_model = RegressionModel(input_size=flattened_size, hidden_size=64).to(device)
    regression_model = RegressionModel(input_size=flattened_size, hidden_sizes=[64,32], activations=['relu', 'relu']).to(device)


    # # Define the optimizer for regression model
    num_epochs = 50
    # initial_lr = 0.0001
    # optimizer = 'Adam'
    # if optimizer == 'Adam':
    #     optimizer = optim.Adam(regression_model.parameters(), lr=initial_lr)
    # elif optimizer == 'SGD':
    #     optimizer = optim.SGD(regression_model.parameters(), lr=initial_lr)
    # elif optimizer == 'RMSprop':
    #     optimizer = optim.RMSprop(regression_model.parameters(), lr=initial_lr)
    # elif optimizer == 'AdamW':
    #     optimizer = optim.AdamW(regression_model.parameters(), lr=initial_lr)
    # else:
    #     raise ValueError("Optimizer type must be one of 'Adam', 'SGD', 'RMSprop', or 'AdamW'")
    # lr_reduction_factor = 0.5       
    # patience_learningRate = 4
    # patience_earlyStop = 13
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=lr_reduction_factor, patience=patience_learningRate, verbose=True)

    # filename = 'encoder_regression'


    # # # Train regression model
    # best_epoch_readable, best_val_loss, best_model_path, run_summary_path, losses_path, train_loss_array, val_loss_array, best_epoch_train_loss, best_epoch_val_loss  = train_model_encoder_regression(model_stationary=encoder, model = regression_model,  num_epochs=num_epochs, train_dataloader=encoder_regression_train_dataloader,
    #                                                                                                     val_dataloader=encoder_regression_val_dataloader, criterion=criterion_autoencoder, optimizer=optimizer,scheduler=scheduler, patience_earlyStop=patience_earlyStop,
    #                                                                                                     filename=filename, model_dir=output_dir)
    # Define model parameter combinations
    model_params = [
        # {'hidden_sizes': [64, 32], 'activations': ['relu', 'relu'], 'use_batch_norm': True, 'use_dropout': True, 'dropout_rate': 0.1},
        # {'hidden_sizes': [64, 32], 'activations': ['relu', 'relu'], 'use_batch_norm': False, 'use_dropout': True, 'dropout_rate': 0.1},
        # {'hidden_sizes': [128, 64], 'activations': ['relu', 'relu'], 'use_batch_norm': True, 'use_dropout': True, 'dropout_rate': 0.1},
        # {'hidden_sizes': [128, 64], 'activations': ['relu', 'relu'], 'use_batch_norm': False, 'use_dropout': True, 'dropout_rate': 0.1},
        # {'hidden_sizes': [64, 32], 'activations': ['tanh', 'tanh'], 'use_batch_norm': True, 'use_dropout': False},
        # {'hidden_sizes': [64, 32], 'activations': ['tanh', 'tanh'], 'use_batch_norm': False, 'use_dropout': False},
        # {'hidden_sizes': [128, 64], 'activations': ['tanh', 'tanh'], 'use_batch_norm': True, 'use_dropout': False},
        # {'hidden_sizes': [32, 16], 'activations': ['relu', 'relu'], 'use_batch_norm': False, 'use_dropout': True, 'dropout_rate': 0.1},
        {'hidden_sizes': [32, 16], 'activations': ['tanh', 'tanh'], 'use_batch_norm': False, 'use_dropout': True, 'dropout_rate': 0.1},
        {'hidden_sizes': [32, 16], 'activations': ['tanh', 'tanh'], 'use_batch_norm': True, 'use_dropout': True, 'dropout_rate': 0.1},
        {'hidden_sizes': [32, 16], 'activations': ['relu', 'tanh'], 'use_batch_norm': False, 'use_dropout': True, 'dropout_rate': 0.1},
        {'hidden_sizes': [32, 16, 16], 'activations': ['tanh', 'tanh', 'tanh'], 'use_batch_norm': False, 'use_dropout': True, 'dropout_rate': 0.1},
        {'hidden_sizes': [32, 32, 16], 'activations': ['tanh', 'tanh', 'tanh'], 'use_batch_norm': True, 'use_dropout': True, 'dropout_rate': 0.1},
        {'hidden_sizes': [32, 16, 8], 'activations': ['tanh', 'tanh', 'tanh'], 'use_batch_norm': False, 'use_dropout': True, 'dropout_rate': 0.1},
        {'hidden_sizes': [64, 32, 16], 'activations': ['tanh', 'tanh','tanh'], 'use_batch_norm': False, 'use_dropout': False}
    ]

    # Define optimizer parameters
    initial_lr = 0.0001
    # optimizer_types = ['Adam', 'RMSprop']
    optimizer_types = ['Adam']


    # Train models with different parameter combinations
    for i, params in enumerate(model_params):
        regression_model = RegressionModel(input_size=flattened_size, **params).to(device)

        for optimizer_type in optimizer_types:
            optimizer = getattr(optim, optimizer_type)(regression_model.parameters(), lr=initial_lr)
            lr_reduction_factor = 0.5       
            patience_learningRate = 4
            patience_earlyStop = 13
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=lr_reduction_factor, patience=patience_learningRate, verbose=True)

            filename = f'encoder_regression_model_2_{i}_optimizer_{optimizer_type}'

            # Train regression model
            best_epoch_readable, best_val_loss, best_model_path, run_summary_path, losses_path, train_loss_array, val_loss_array, best_epoch_train_loss, best_epoch_val_loss = train_model_encoder_regression(model_stationary=encoder, model=regression_model, num_epochs=num_epochs, train_dataloader=encoder_regression_train_dataloader, val_dataloader=encoder_regression_val_dataloader, criterion=criterion_autoencoder, optimizer=optimizer, scheduler=scheduler, patience_earlyStop=patience_earlyStop, filename=filename, model_dir=output_dir, model_params=params, optimizer_type=optimizer_type)

if __name__ in "__main__":
    main()
