from classifiers_util import *
from lstm_pulseNum_classifier import CustomLSTMClassifier
# Get the directory of the currently running file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the utils directory relative to the current file's directory
utils_dir = os.path.abspath(os.path.join(current_dir, '../..', 'ml_backbone'))
denoise_dir = os.path.abspath(os.path.join(current_dir, '../..', 'denoising'))

sys.path.append(utils_dir)
sys.path.append(denoise_dir)


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
    # datapath = "/Users/jhirschm/Documents/MRCO/Data_Changed/Test"
    datapath1 = "/sdf/data/lcls/ds/prj/prjs2e21/results/1-Pulse_03282024/Processed_06252024/"
    datapath2 = "/sdf/data/lcls/ds/prj/prjs2e21/results/even-dist_Pulses_03302024/Processed_06252024/"
    datapath_train = "/sdf/data/lcls/ds/prj/prjs2e21/results/even-dist_Pulses_03302024/Processed_07262024/train/"
    datapath_val = "/sdf/data/lcls/ds/prj/prjs2e21/results/even-dist_Pulses_03302024/Processed_07262024/val/"
    datapath_test = "/sdf/data/lcls/ds/prj/prjs2e21/results/even-dist_Pulses_03302024/Processed_07262024_0to1/test/"
    datapath_test = "/sdf/scratch/lcls/ds/prj/prjs2e21/scratch/fast_data_access/even-dist_Pulses_03302024/Processed_07262024_0to1/train/test/"
    # datapath_test = "/sdf/data/lcls/ds/prj/prjs2e21/results/2-Pulse_04232024/Processed_07312024_0to1/test/"

    pulse_specification = None


    # data = DataMilking_Nonfat(root_dir=datapath, pulse_number=2, subset=4)
    # data = DataMilking_SemiSkimmed(root_dir=datapath, pulse_number=1, input_name="Ximg", labels=["Ypdf"])
    # data_test = DataMilking_MilkCurds(root_dirs=[datapath_test], input_name="Ypdf", pulse_handler=None, transform=None, pulse_threshold=4, test_batch=1, zero_to_one_rescale=True)
    # data_test = DataMilking_MilkCurds(root_dirs=[datapath_test], input_name="Ximg", pulse_handler=None, transform=None, pulse_threshold=4, test_batch=1, zero_to_one_rescale=False)
   
    # data_test = DataMilking_MilkCurds(root_dirs=[datapath_test], input_name="Ximg", pulse_handler=None, transform=None, pulse_threshold=4, zero_to_one_rescale=False, test_batch=1)
    data_test = DataMilking_MilkCurds(root_dirs=[datapath_test], input_name="Ximg", pulse_handler=None, transform=None, pulse_threshold=4, zero_to_one_rescale=False, phases_labeled=False, phases_labeled_max=2)



    # data_val = DataMilking_MilkCurds(root_dirs=[datapath_val], input_name="Ypdf", pulse_handler=None, transform=None, pulse_threshold=4, test_batch=3)

    # Calculate the lengths for each split
    train_size = 0
    val_size = 0
    test_size = len(data_test) - train_size - val_size
    #print sizes of train, val, and test
    print(f"Train size: {train_size}")
    print(f"Validation size: {val_size}")
    print(f"Test size: {test_size}")

    # Perform the split
    train_dataset, val_dataset, test_dataset = random_split(data_test, [train_size, val_size, test_size])


    # Create data loaders
    # train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)


    # Define the model
    # Create CustomLSTMClassifier model
    # data = {
    #     "hidden_size": 64,
    #     "num_lstm_layers": 3,
    #     "bidirectional": True,
    #     "fc_layers": [32, 64],
    #     "dropout": 0.2,
    #     "lstm_dropout": 0.2,
    #     "layerNorm": False,
    #     # Other parameters are default or not provided in the example
    # }   
    data = {
        "hidden_size": 128,#128
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
    num_classes = 5   # Example number of classes

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
        ignore_output_layer=False  # Set as needed based on your application
    )
    classModel.to(device)

        # model_save_dir = "/Users/jhirschm/Documents/MRCO/Data_Changed/Test"
    model_save_dir = "/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/lstm_classifier/run_073312024_5classCase/evalOutput_5classCase_temp"
    model_save_dir = "/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/lstm_classifier/run_09052024_5classCase/evalOutput_5classCase_10092024"
    # Check if directory exists, otherwise create it
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)




    # best_autoencoder_model_path = "/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/denoising/run_06272024_singlePulse/testAutoencoder_best_model.pth"
    # best_autoencoder_model_path = "/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/denoising/run_07272024_singlePulse_2/autoencoder_best_model.pth"
    best_autoencoder_model_path = "/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/denoising/run_07282024_multiPulse/autoencoder_best_model.pth"
    best_autoencoder_model_path = "/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/denoising/run_09042024_multiPulse_final/autoencoder_6_best_model.pth"

    
    best_model_zero_mask_path = "/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/denoising/run_07272024_zeroPredict/classifier_best_model.pth"

    # best_model_zero_mask_path = "/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/denoising/run_07042024_zeroPredict/classifier_best_model.pth"
    # best_model_zero_mask_path = "/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/denoising/run_07272024_zeroPredict/classifier_best_model.pth"

    # best_mode_classifier = "/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/lstm_classifier/run_07262024_ypdf_0to1_4/testLSTM_best_model.pth"
    # best_mode_classifier = "/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/lstm_classifier/run_07262024_ypdf_0to1_5_extraClass/testLSTM_best_model.pth"
    # best_mode_classifier = "/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/lstm_classifier/run_07302024_ypdf_0to1_test3/testLSTM_best_model.pth"
    best_mode_classifier = "/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/lstm_classifier/run_073312024_5classCase/testLSTM_best_model.pth"
    best_mode_classifier = "/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/lstm_classifier/run_09052024_5classCase/testLSTM_XimgDenoised_best_model.pth"
   
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
    encoder_layers = np.array([
        [nn.Conv2d(1, 16, kernel_size=3, padding=2), nn.ReLU()],
        [nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU()],
        [nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU()]])
   
    # decoder_layers = np.array([
    #     [nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1), nn.ReLU()],
    #     [nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1), nn.ReLU()],
    #     [nn.ConvTranspose2d(16, 1, kernel_size=3, padding=2), nn.Tanh()]  # Example with Sigmoid activation
    #     # [nn.ConvTranspose2d(16, 1, kernel_size=3, padding=2), None],  # Example without activation
    # ])
    decoder_layers = np.array([
        [nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1), nn.ReLU()],
        [nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1), nn.ReLU()],
        [nn.ConvTranspose2d(16, 1, kernel_size=3, padding=2), nn.Sigmoid()]  # Example with Sigmoid activation
        # [nn.ConvTranspose2d(16, 1, kernel_size=3, padding=2), None],  # Example without activation
    ])
    
    autoencoder = Ximg_to_Ypdf_Autoencoder(encoder_layers, decoder_layers)

    # Example usage
    conv_layers = [
        [nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1), nn.ReLU()],
        [nn.MaxPool2d(kernel_size=2, stride=2, padding=0), None],
        [nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), nn.ReLU()],
        [nn.MaxPool2d(kernel_size=2, stride=2, padding=0), None]
    ]

    # Calculate the output size after conv layers
    def get_conv_output_size(input_size, conv_layers):
        x = torch.randn(input_size)
        model = nn.Sequential(*[layer for layer_pair in conv_layers for layer in layer_pair if layer is not None])
        x = model(x)
        return x.shape

    output_size = get_conv_output_size((1, 1, 512, 16), conv_layers)
    print(f"Output size after conv layers: {output_size}")

    # Use the calculated size for the fully connected layer input
    fc_layers = [
        [nn.Linear(output_size[1] * output_size[2] * output_size[3], 4), nn.ReLU()],
        [nn.Linear(4, 1), None]
    ]

    zero_model = Zero_PulseClassifier(conv_layers, fc_layers)
    autoencoder.to(device)
    state_dict = torch.load(best_autoencoder_model_path, map_location=device)
    autoencoder.load_state_dict(state_dict)

    zero_model.to(device)
    state_dict = torch.load(best_model_zero_mask_path, map_location=device)
    print(state_dict.keys())
    # Remove keys related to side_network
    keys_to_remove = ['side_network.0.weight', 'side_network.0.bias']
    state_dict = {k: v for k, v in state_dict.items() if not any(key in k for key in keys_to_remove)}


    zero_model.load_state_dict(state_dict)

    identifier = "testLSTM"
    autoencoder.to(device)
    zero_model.to(device)

    # Calculate the total number of parameters
    total_params = sum(param.numel() for param in classModel.parameters())

    print(f'Total number of parameters: {total_params}')
    classModel.evaluate_model(test_dataloader, identifier, model_save_dir, device, denoising=True, denoise_model = autoencoder, zero_mask_model = zero_model, two_pulse_analysis=False, two_and_three_pulse_store=True)
    
    
    # results_file = os.path.join(model_save_dir, f"{identifier}_results.txt")
    # with open(results_file, 'w') as f:
    #     f.write("Model Training Results\n")
    #     f.write("======================\n")
    #     f.write(f"Data Path: {datapath2}\n")
    #     f.write(f"Model Save Directory: {model_save_dir}\n")
    #     f.write("\nModel Parameters and Hyperparameters\n")
    #     f.write("-----------------------------------\n")
    #     f.write(f"Patience: {scheduler.patience}\n")
    #     f.write(f"Cooldown: {scheduler.cooldown}\n")
    #     f.write(f"Learning Rate Reduction Factor: {scheduler.lr_reduction_factor}\n")
    #     f.write(f"Improvement Percentage: {scheduler.improvement_percentage}\n")
    #     f.write(f"Initial Learning Rate: {optimizer.param_groups[0]['lr']}\n")
    #     f.write("\nModel Architecture\n")
    #     f.write("------------------\n")
    #     f.write(f"Encoder Layers: {encoder_layers}\n")
    #     f.write(f"Decoder Layers: {decoder_layers}\n")
    #     f.write("\nAdditional Notes\n")
    #     f.write("----------------\n")
    #     f.write("First trial on S3DF for LSTM trained on denoised data.\n")

    
    
if __name__ == "__main__":
    main()