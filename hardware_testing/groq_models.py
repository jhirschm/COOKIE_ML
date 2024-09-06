import os
from groqflow import groqit
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the utils directory relative to the current file's directory
utils_dir = os.path.abspath(os.path.join(current_dir, '../src/', 'ml_backbone'))
denoise_dir = os.path.abspath(os.path.join(current_dir, '../src/', 'denoising'))

sys.path.append(utils_dir)
sys.path.append(denoise_dir)

from denoising_util import *
from ximg_to_ypdf_autoencoder import Ximg_to_Ypdf_Autoencoder, Zero_PulseClassifier

# Example usage
encoder_layers = np.array([
    [nn.Conv2d(1, 16, kernel_size=3, padding=2), nn.ReLU()],
    [nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU()],
    [nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU()]])

decoder_layers = np.array([
    [nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1), nn.ReLU()],
    [nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1), nn.ReLU()],
    [nn.ConvTranspose2d(16, 1, kernel_size=3, padding=2), nn.Sigmoid()]  # Example with Sigmoid activation
    # [nn.ConvTranspose2d(16, 1, kernel_size=3, padding=2), None],  # Example without activation
])


autoencoder = Ximg_to_Ypdf_Autoencoder(encoder_layers, decoder_layers)

inputs = {"x": torch.rand(1, 1, 512, 16)} 
gmodel = groqit(autoencoder, inputs, groqview=True, rebuild="always", build_name="model "+str(1))
gmodel.groqview()
  # Get performance estimates in terms of latency and throughput
estimate = gmodel.estimate_performance()
print("Your build's estimated performance is:")
print(f"{estimate.latency:.7f} {estimate.latency_units}")
print(f"{estimate.throughput:.1f} {estimate.throughput_units}")
print("Example estimate_performance.py finished")
