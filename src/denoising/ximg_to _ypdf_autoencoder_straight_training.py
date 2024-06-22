from ximg_to_ypdf_autoencoder import Ximg_to_Ypdf_Autoencoder
from denoising_util import *
# Get the directory of the currently running file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the utils directory relative to the current file's directory
utils_dir = os.path.abspath(os.path.join(current_dir, '..', 'ml_backbone'))

# Add the utils directory to the Python path
sys.path.append(utils_dir)
from utils import DataMilking_Nonfat, DataMilking
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
device = torch.device("cpu")
def main():
    # Input Data Paths and Output Save Paths

    # Load Dataset and Feed to Dataloader
    datapath = "/Users/jhirschm/Documents/MRCO/Data_Changed/Test"
    dataset = DataMilking(root_dir=datapath, attributes=["energies", "phases", "npulses"], pulse_number=2)

    print(dataset)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    data = DataMilking_Nonfat(root_dir=datapath, pulse_number=2)
    train_dataloader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True) #need to fix eventually
    val_dataloader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=False)


    # Define the model
    encoder_layers = [
        nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
        nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
    ]

    decoder_layers = [
        nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
    ]

    autoencoder = Ximg_to_Ypdf_Autoencoder(encoder_layers, decoder_layers)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
    scheduler = CustomScheduler(optimizer, patience=5, cooldown=2, lr_reduction_factor=0.1, min_lr=1e-6, improvement_percentage=0.01)

    
    model_save_dir = "/Users/jhirschm/Documents/MRCO/Data_Changed/Test"
    identifier = "testAutoencoder"
    autoencoder.train_model(train_dataloader, val_dataloader, criterion, optimizer, scheduler, model_save_dir, identifier, device, checkpoints_enabled=True, resume_from_checkpoint=False, max_epochs=20)


    # Train the model
    
if __name__ == "__main__":
    main()