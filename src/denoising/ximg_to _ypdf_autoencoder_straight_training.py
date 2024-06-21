from ximg_to_ypdf_autoencoder import Ximg_to_Ypdf_Autoencoder
from denoising_util import *
import utils.DataMilking as DataMilking


def main():
    # Input Data Paths and Output Save Paths

    # Load Dataset and Feed to Dataloader
    data = DataMilking(root_dir="data", img_type="Ypdf", attributes=["npulses"], pulse_range=[10, 20], transform=transforms.ToTensor())
    dataloader = torch.utils.data.DataLoader(data, batch_size=4, shuffle=True)

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

    # Train the model
    
   