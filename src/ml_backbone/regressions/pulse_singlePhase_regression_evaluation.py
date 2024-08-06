from regression_util import *
from pulse_phase_regression import RegressionModel



# Get the directory of the currently running file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the utils directory relative to the current file's directory
utils_dir = os.path.abspath(os.path.join(current_dir, '../..', 'ml_backbone'))
classifier_dir = os.path.abspath(os.path.join(current_dir, '../../ml_backbone', 'classifiers'))

denoise_dir = os.path.abspath(os.path.join(current_dir, '../..', 'denoising'))

sys.path.append(utils_dir)
sys.path.append(denoise_dir)
sys.path.append(classifier_dir)
from lstm_pulseNum_classifier import CustomLSTMClassifier
from ximg_to_ypdf_autoencoder import Ximg_to_Ypdf_Autoencoder, Zero_PulseClassifier

from utils import DataMilking_Nonfat, DataMilking, DataMilking_SemiSkimmed, DataMilking_HalfAndHalf, DataMilking_MilkCurds
from utils import CustomScheduler

# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    print("GPU is available!")
    device = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
    print("MPS is available. Using GPU.")
else:
    device = torch.device("cpu")
    print("MPS is not available. Using CPU.")
# device = torch.device("cpu")
def main():
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Input Data Paths and Output Save Paths

    # Load Dataset and Feed to Dataloader
    datapath_test = "/sdf/data/lcls/ds/prj/prjs2e21/results/1-Pulse_03282024/Processed_07262024_0to1/test/"

    pulse_specification = None


    data_train = DataMilking_MilkCurds(root_dirs=[datapath_test], input_name="Ypdf", pulse_handler=None, transform=None, pulse_threshold=4, zero_to_one_rescale=False, test_batch =2, phases_labeled=True, phases_labeled_max=1)
    # data_train = DataMilking_HalfAndHalf(root_dirs=[datapath_train], input_name="Ypdf", labels = ["phases"], pulse_handler = None, transform=None, test_batch=2)
    print(len(data_train))
    # Calculate the lengths for each split
    train_size = int(0.8 * len(data_train))
    val_size = int(0.2 * len(data_train))
    test_size = len(data_train) - train_size - val_size
    #print sizes of train, val, and test
    print(f"Train size: {train_size}")
    print(f"Validation size: {val_size}")
    print(f"Test size: {test_size}")

    # Perform the split
    train_dataset, val_dataset, test_dataset = random_split(data_train, [train_size, val_size, test_size])



    # Create data loaders
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model_save_dir = "/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/lstm_classifier/run_08052024_regressionSingleLSTMTest_4/evaluation"
    # Check if directory exists, otherwise create it
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    
    best_autoencoder_model_path = "/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/denoising/run_07282024_multiPulse/autoencoder_best_model.pth"
    best_model_zero_mask_path = "/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/denoising/run_07272024_zeroPredict/classifier_best_model.pth"
    # best_mode_classifier = "/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/lstm_classifier/run_07302024_ypdf_0to1_test3/testLSTM_best_model.pth"
    best_mode_classifier = "/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/lstm_classifier/run_073312024_5classCase/testLSTM_best_model.pth"
    best_model_regression_path = "/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/lstm_classifier/run_08052024_regressionSingleLSTMTest_4/regression_model_best_model.pth"
    # Create LSTM and Denoiser Network that data will go through first before reaching regression network
    # Define the model
    # Create CustomLSTMClassifier model
    data = {
        "hidden_size": 128,
        "num_lstm_layers": 3,
        "bidirectional": True,
        "fc_layers": [32, 64],
        "dropout": 0.2,
        "lstm_dropout": 0.2,
        "layerNorm": False,
        # Other parameters are default or not provided in the example
    }   

    # Assuming input_size and num_classes are defined elsewhere
    input_size = 512  # Define your input size
    num_classes = 6   # Example number of classes

    # Instantiate the CustomLSTMClassifier
    classModel = CustomLSTMClassifier(
        input_size=input_size,
        hidden_size=data['hidden_size'],
        num_lstm_layers=data['num_lstm_layers'],
        num_classes=num_classes,
        bidirectional=data['bidirectional'],
        fc_layers=data['fc_layers'],
        dropout_p=data['dropout'],
        lstm_dropout=data['lstm_dropout'],
        layer_norm=data['layerNorm'],
        ignore_output_layer=False,  # Set as needed based on your application
        ignore_fc_layers=True
    )

    classModel.to(device)
    state_dict = torch.load(best_mode_classifier, map_location=device)
    def remove_module_prefix(state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        return new_state_dict
    state_dict = remove_module_prefix(state_dict)
    for key in state_dict.keys():
        print(key, state_dict[key].shape)
    classModel.load_state_dict(state_dict)

    

     # Example usage
    conv_layers = [
        [nn.Conv2d(1, 16, kernel_size=3, padding=2), nn.ReLU()],
        [nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU()],
        [nn.MaxPool2d(kernel_size=2, stride=2), None],
        [nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU()],
        [nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU()],
        [nn.MaxPool2d(kernel_size=2, stride=2), None]
        ]
   
    conv_output_size = get_conv_output_size((1, 1, 512, 16), conv_layers)
    conv_output_size_flattened = conv_output_size[1] * conv_output_size[2] * conv_output_size[3]
    print(f"Output size after conv layers: {conv_output_size_flattened}")

    #Trying LSTM 
    fc_layers = [
    [nn.Linear(conv_output_size_flattened, 256), nn.ReLU()],
    [nn.Linear(256, 128), nn.ReLU()],
    [nn.Linear(128, 32), nn.ReLU()],
    [nn.Linear(32, 8), nn.ReLU()],
    # [nn.Linear(32, 8), nn.ReLU()],
    [nn.Linear(8, 1), nn.ReLU()]  
    ]

    # Define LSTM configuration
    lstm_config = {
        'input_size': 512,  # Example input size for LSTM
        'hidden_size': 256,
        'num_layers': 2,
        'bidirectional': False
    }

    # Create the RegressionModel instance
    regression_model = RegressionModel(fc_layers=fc_layers, 
                            conv_layers=conv_layers,  # No convolutional layers
                            lstm_config=None, 
                            dtype=torch.float32, 
                            use_dropout=False, 
                            dropout_rate=0.1)
    regression_model.to(device)
    state_dict = torch.load(best_model_regression_path, map_location=device)
    state_dict = remove_module_prefix(state_dict)
    for key in state_dict.keys():
        print(key, state_dict[key].shape)
    regression_model.load_state_dict(state_dict)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    
    max_epochs = 200

    
    identifier = "regression_model"
    regression_model.evaluate_model(test_dataloader, criterion, model_save_dir, identifier, device, denoising=False, 
                                denoise_model =None , zero_mask_model = None, lstm_pretrained_model = None, parallel=True, single_pulse=True, single_pulse_analysis=True)
    print(summary(model=regression_model, 
        input_size=(32, 16, 512), # make sure this is "input_size", not "input_shape"
        # col_names=["input_size"], # uncomment for smaller output
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
))
    
    # results_file = os.path.join(model_save_dir, f"{identifier}_results.txt")
    # with open(results_file, 'w') as f:
    #     f.write("Model Training Results\n")
    #     f.write("======================\n")
    #     f.write(f"Data Path: {datapath_test}\n")
    #     f.write(f"Model Save Directory: {model_save_dir}\n")
    #     f.write("\nData Settings\n")
    #     f.write("-------------\n")
    #     f.write(f"Data Path: {datapath_train}\n")
    #     f.write(f"Pulse Specification: {pulse_specification}\n")
    #     f.write("\nModel Parameters and Hyperparameters\n")
    #     f.write("-----------------------------------\n")
    #     f.write(f"Patience: {scheduler.patience}\n")
    #     f.write(f"Cooldown: {scheduler.cooldown}\n")
    #     f.write(f"Learning Rate Reduction Factor: {scheduler.lr_reduction_factor}\n")
    #     f.write(f"Improvement Percentage: {scheduler.improvement_percentage}\n")
    #     f.write(f"Initial Learning Rate: {optimizer.param_groups[0]['lr']}\n")
    #     f.write(f"Max Epochs: {max_epochs}\n")
    #     f.write("\nModel Architecture\n")
    #     f.write("------------------\n")

    #     f.write("Denoising Model:\n")
    #     f.write("  Encoder Layers:\n")
    #     for layer in encoder_layers:
    #         f.write(f"    {layer}\n")
    #     f.write("  Decoder Layers:\n")
    #     for layer in decoder_layers:
    #         f.write(f"    {layer}\n")
    #     f.write(f"  Model Path: {best_autoencoder_model_path}\n")
        
    #     f.write("\nZero Pulse Classifier Model:\n")
    #     f.write("  Convolutional Layers:\n")
    #     for layer in conv_layers:
    #         f.write(f"    {layer}\n")
    #     f.write("  Fully Connected Layers:\n")
    #     for layer in fc_layers:
    #         f.write(f"    {layer}\n")
    #     f.write(f"  Model Path: {best_model_zero_mask_path}\n")

    #     f.write("\nLSTM Classifier Model:\n")
    #     for key, value in data.items():
    #         f.write(f"  {key}: {value}\n")
    #     f.write(f"  Model Path: {best_mode_classifier}\n")
    #     f.write(f"  Removed FC Layers\n")
        
    #     f.write("\nRegression Model:\n")
    #     f.write("  Fully Connected Layers:\n")
    #     for layer, activation in fc_layers:
    #         f.write(f"    Layer: {layer}\n")
    #         if activation is not None:
    #             f.write(f"    Activation: {activation}\n")

    #     f.write("\nAdditional Notes\n")
    #     f.write("----------------\n")
    #     f.write("LSTM trained on YPDF making sure images between 0 and 1 (instead of -1 to 1). No denoising on Ypdf. Denoising Ximg.\n")


        
if __name__ == "__main__":
    main()
