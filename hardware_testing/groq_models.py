import os
import sys
from groqflow import groqit
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the utils directory relative to the current file's directory
utils_dir = os.path.abspath(os.path.join(current_dir, '../src/', 'ml_backbone'))
denoise_dir = os.path.abspath(os.path.join(current_dir, '../src/', 'denoising'))
class_dir = os.path.abspath(os.path.join(current_dir, '../src/ml_backbone', 'classifiers'))


sys.path.append(utils_dir)
sys.path.append(denoise_dir)
sys.path.append(class_dir)


from denoising_util import *
from ximg_to_ypdf_autoencoder import Ximg_to_Ypdf_Autoencoder, Zero_PulseClassifier

from classifiers_util import *
from lstm_pulseNum_classifier import CustomLSTMClassifier
# Get the directory of the currently running file

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
gmodel = groqit(autoencoder, inputs, groqview=True, rebuild="always", build_name="model_denoise_autoencoder")
  # Get performance estimates in terms of latency and throughput
estimate = gmodel.estimate_performance()
print("Your build's estimated performance is:")
print(f"{estimate.latency:.7f} {estimate.latency_units}")
print(f"{estimate.throughput:.1f} {estimate.throughput_units}")
print("Example estimate_performance.py finished")

# Assuming input_size and num_classes are defined elsewhere
input_size = 512  # Define your input size
num_classes = 5   # Example number of classes
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
inputs = {"x": torch.rand(1, 16, 512)} 

gmodel = groqit(classModel, inputs, groqview=True, rebuild="always", build_name="model_classifier_lstm")
estimate = gmodel.estimate_performance()

print("Your build's estimated performance is:")
print(f"{estimate.latency:.7f} {estimate.latency_units}")
print(f"{estimate.throughput:.1f} {estimate.throughput_units}")
print("Example estimate_performance.py finished")

# gmodel.groqview()



