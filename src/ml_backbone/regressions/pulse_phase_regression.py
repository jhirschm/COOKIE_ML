from regression_util import *

    
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
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')

        # Initialize biases to zeros
        for layer in self.hidden_layers:
            if isinstance(layer, nn.Linear):
                nn.init.constant_(layer.bias, 0)
        nn.init.constant_(self.output_layer.bias, 0)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x


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
