from classifiers_util import *
from lstm_pulseNum_classifier import CustomLSTMClassifier
# Get the directory of the currently running file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the utils directory relative to the current file's directory
utils_dir = os.path.abspath(os.path.join(current_dir, '../..', 'ml_backbone'))
denoise_dir = os.path.abspath(os.path.join(current_dir, '../..', 'denoising'))

sys.path.append(utils_dir)
sys.path.append(denoise_dir)
from ximg_to_ypdf_autoencoder import Zero_PulseClassifier


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

def main():
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Input Data Paths and Output Save Paths
    
    datapath_test = "/sdf/data/lcls/ds/prj/prjs2e21/results/even-dist_Pulses_03302024/Processed_07262024_0to1/test/"
    datapaths = [datapath_test]


    
    data = DataMilking_MilkCurds(root_dirs=datapaths, input_name="Ximg", pulse_handler=None, transform=None, pulse_threshold=None, pulse_min_binary=4, test_batch=1)
    # Calculate the lengths for each split    # data_val = DataMilking_MilkCurds(root_dirs=[datapath_val], input_name="Ypdf", pulse_handler=None, transform=None, pulse_threshold=4, test_batch=3)

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

    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)


    


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

    classifier = Zero_PulseClassifier(conv_layers, fc_layers)
    best_model = "/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/lstm_classifier/run_07292024_highPulseCountClassifier/classifier_best_model.pth"
    classifier.to(device)
    state_dict = torch.load(best_model, map_location=device)
    print(state_dict.keys())
    # Remove keys related to side_network
    keys_to_remove = ['side_network.0.weight', 'side_network.0.bias']
    state_dict = {k: v for k, v in state_dict.items() if not any(key in k for key in keys_to_remove)}


    classifier.load_state_dict(state_dict)
    classifier.to(device)

    # Define the loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy with Logits Loss
    # criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    max_epochs = 200
    scheduler = CustomScheduler(optimizer, patience=3, early_stop_patience = 8, cooldown=2, lr_reduction_factor=0.5, max_num_epochs = max_epochs, improvement_percentage=0.001)
    # model_save_dir = "/Users/jhirschm/Documents/MRCO/Data_Changed/Test"
    model_save_dir = "/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/lstm_classifier/run_07292024_highPulseCountClassifier/evalOutputs/"
    # Check if directory exists, otherwise create it
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    identifier = "classifier"
    classifier.to(device)
    classifier.evaluate_model(test_dataloader,  model_save_dir = model_save_dir, identifier = identifier, device=device)
    
if __name__ == "__main__":
    main()