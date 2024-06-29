from ximg_to_ypdf_autoencoder import Ximg_to_Ypdf_Autoencoder
from denoising_util import *
# Get the directory of the currently running file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the utils directory relative to the current file's directory
utils_dir = os.path.abspath(os.path.join(current_dir, '..', 'ml_backbone'))

# Add the utils directory to the Python path
sys.path.append(utils_dir)
from utils import DataMilking_Nonfat, DataMilking, DataMilking_SemiSkimmed
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
    # datapath = "/sdf/data/lcls/ds/prj/prjs2e21/results/2-Pulse_04232024/Processed_06212024/"
    datapath = "/sdf/data/lcls/ds/prj/prjs2e21/results/even-dist_Pulses_03302024/Processed_06252024/"
    # datapath = "/sdf/data/lcls/ds/prj/prjs2e21/results/1-Pulse_03282024/Processed_06252024/"
    # dataset = DataMilking(root_dir=datapath, attributes=["energies", "phases", "npulses"], pulse_number=2)


    data = DataMilking_SemiSkimmed(root_dir=datapath, pulse_number_max=10, input_name="Ximg", labels=["Ypdf"], test_batch=2)
    # Calculate the lengths for each split
    train_size = int(0.8 * len(data))
    val_size = int(0.1 * len(data))
    test_size = int(len(data) - train_size - val_size)

    # Perform the split
    train_dataset, val_dataset, test_dataset = random_split(data, [train_size, val_size, test_size])

     # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8)


    # Example usage
    encoder_layers = np.array([
        [nn.Conv2d(1, 16, kernel_size=3, padding=2), nn.ReLU()],
        [nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU()],
        [nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU()]])
   
    decoder_layers = np.array([
        [nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1), nn.ReLU()],
        [nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1), nn.ReLU()],
        [nn.ConvTranspose2d(16, 1, kernel_size=3, padding=2), nn.Tanh()]  # Example with Sigmoid activation
        # [nn.ConvTranspose2d(16, 1, kernel_size=3, padding=2), None],  # Example without activation
    ])
    
    autoencoder = Ximg_to_Ypdf_Autoencoder(encoder_layers, decoder_layers)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    # model_save_dir = "/Users/jhirschm/Documents/MRCO/Data_Changed/Test"
    model_save_dir = "/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/denoising/run_06262024_singlePulse/outputs_fromEvenDist_max10Pulses_fineTuned0Pulse"
    # best_model_path = "/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/denoising/run_06262024_singlePulse/testAutoencoder_best_model.pth"
    best_model_path = "/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/denoising/run_06272024_singlePulse_fineTuned0Pulse/testAutoencoder_fineTuning0Pulse_fine_tuned_best_model.pth"

   
    autoencoder.to(device)
    state_dict = torch.load(best_model_path, map_location=device)
    autoencoder.load_state_dict(state_dict)
    

    # Check if directory exists, otherwise create it
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    identifier = "testAutoencoder_eval"
    autoencoder.evaluate_model(test_dataloader, criterion, device, save_results=True, results_dir=model_save_dir, results_filename=f"{identifier}_results.h5")
    results_file = os.path.join(model_save_dir, f"{identifier}_results.txt")
    with open(results_file, 'w') as f:
        f.write("Model Training Results\n")
        f.write("======================\n")
        f.write(f"Data Path: {datapath}\n")
        f.write(f"Model Save Directory: {model_save_dir}\n")
        f.write("\nModel Parameters and Hyperparameters\n")
        f.write("-----------------------------------\n")
        f.write("\nModel Architecture\n")
        f.write("------------------\n")
        f.write(f"Encoder Layers: {encoder_layers}\n")
        f.write(f"Decoder Layers: {decoder_layers}\n")
        f.write("\nAdditional Notes\n")
        f.write("----------------\n")
        f.write("Results for inspection on test. Running on even pulses but trained on 1 pulse. Max 10 pulses\n")

    
    
if __name__ == "__main__":
    main()