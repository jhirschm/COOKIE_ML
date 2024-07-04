from ximg_to_ypdf_autoencoder import Ximg_to_Ypdf_Autoencoder
from ximg_to_ypdf_autoencoder import Zero_PulseClassifier


from denoising_util import *

# Get the directory of the currently running file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the utils directory relative to the current file's directory
utils_dir = os.path.abspath(os.path.join(current_dir, '..', 'ml_backbone'))

# Add the utils directory to the Python path
sys.path.append(utils_dir)
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

    # Load Dataset and Feed to Dataloader
    # datapath = "/Users/jhirschm/Documents/MRCO/Data_Changed/Test"
    # datapath = "/sdf/data/lcls/ds/prj/prjs2e21/results/2-Pulse_04232024/Processed_06212024/"
    datapath1 = "/sdf/data/lcls/ds/prj/prjs2e21/results/1-Pulse_03282024/Processed_06252024/"
    datapath2 = "/sdf/data/lcls/ds/prj/prjs2e21/results/even-dist_Pulses_03302024/Processed_06252024/"
    datapaths = [datapath2]
    pulse_specification = None


    # data = DataMilking_Nonfat(root_dir=datapath, pulse_number=2, subset=4)
    # data = DataMilking_SemiSkimmed(root_dir=datapath, pulse_number=1, input_name="Ximg", labels=["Ypdf"])
    data = DataMilking_MilkCurds(root_dirs=datapaths, input_name="Ximg", pulse_handler=None, transform=None, test_batch=2, pulse_threshold=1)
    print(len(data))
    # Calculate the lengths for each split
    train_size = int(0.8 * len(data))
    val_size = int(0.1 * len(data))
    test_size = len(data) - train_size - val_size
    #print sizes of train, val, and test
    print(f"Train size: {train_size}")
    print(f"Validation size: {val_size}")
    print(f"Test size: {test_size}")

    # Perform the split
    train_dataset, val_dataset, test_dataset = random_split(data, [train_size, val_size, test_size])

    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8)


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
        [nn.Linear(output_size[1] * output_size[2] * output_size[3], 128), nn.ReLU()],
        [nn.Linear(128, 1), None]
    ]

    classifier = Zero_PulseClassifier(conv_layers, fc_layers)

    # Define the loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy with Logits Loss
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.1)
    max_epochs = 200
    scheduler = CustomScheduler(optimizer, patience=3, early_stop_patience = 8, cooldown=2, lr_reduction_factor=0.5, max_num_epochs = max_epochs, improvement_percentage=0.001)
    # model_save_dir = "/Users/jhirschm/Documents/MRCO/Data_Changed/Test"
    model_save_dir = "/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/denoising/run_07042024_zeroPredictTest/"
    # Check if directory exists, otherwise create it
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    identifier = "classifier"
    classifier.to(device)
    classifier.train_model(train_dataloader, val_dataloader, criterion, optimizer, scheduler, model_save_dir, identifier, device, checkpoints_enabled=True, resume_from_checkpoint=False, max_epochs=max_epochs)

    results_file = os.path.join(model_save_dir, f"{identifier}_results.txt")
    with open(results_file, 'w') as f:
        f.write("Model Training Results\n")
        f.write("======================\n")
        f.write(f"Data Path: {datapaths}\n")
        f.write(f"Model Save Directory: {model_save_dir}\n")
        f.write("\nModel Parameters and Hyperparameters\n")
        f.write("-----------------------------------\n")
        f.write(f"Patience: {scheduler.patience}\n")
        f.write(f"Cooldown: {scheduler.cooldown}\n")
        f.write(f"Learning Rate Reduction Factor: {scheduler.lr_reduction_factor}\n")
        f.write(f"Improvement Percentage: {scheduler.improvement_percentage}\n")
        f.write(f"Initial Learning Rate: {optimizer.param_groups[0]['lr']}\n")
        f.write("\nModel Architecture\n")
        f.write("------------------\n")
        f.write("\nAdditional Notes\n")
        f.write("----------------\n")
        f.write("Training on even data. Any number of pulses. One hot encoding for 0 or 1+\n")


    
if __name__ == "__main__":

    main()