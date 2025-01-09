from ximg_to_ypdf_autoencoder import Ximg_to_Ypdf_Autoencoder
from ximg_to_ypdf_autoencoder import Zero_PulseClassifier
import torch
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torchinfo import summary
# Get the directory of the currently running file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the utils directory relative to the current file's directory
utils_dir = os.path.abspath(os.path.join(current_dir, '..', 'ml_backbone'))






# device = torch.device("cpu")
def main():
    torch.manual_seed(0)
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

    # Example usage
    conv_layers = [
        [nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1), nn.ReLU()],
        [nn.MaxPool2d(kernel_size=2, stride=2, padding=0), None],
        [nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), nn.ReLU()],
        [nn.MaxPool2d(kernel_size=2, stride=2, padding=0), None]
    ]

    def get_conv_output_size(input_size, conv_layers):
        x = torch.randn(*input_size)  # Ensure input_size is unpacked properly
        print(f"Input shape: {x.shape}")
        with torch.no_grad():
            model = nn.Sequential(
                *[layer for layer_pair in conv_layers for layer in layer_pair if layer is not None]
            )
            for i, layer in enumerate(model):
                x = layer(x)
                print(f"Shape after layer {i}: {x.shape}")
        return x.shape

    output_size = get_conv_output_size((1, 1, 512, 16), conv_layers)
    # print(f"Output size after conv layers: {output_size}")

    # Use the calculated size for the fully connected layer input
    # fc_layers = [
    #     [nn.Linear(output_size[1] * output_size[2] * output_size[3], 4), nn.ReLU()],
    #     [nn.Linear(4, 1), None]
    # ]

    # classifier = Zero_PulseClassifier(conv_layers, fc_layers)

    # batch_size = 1
    # input_channels = 1
    # height, width = 16, 512
    # inputs = {"x": torch.randn(batch_size, input_channels, height, width)}

    # summary_file = "~/Downloads/zero_pulse_classifier.txt"

    # # Redirect stdout to the file
    # with open(summary_file, "w") as f:
    #     sys.stdout = f
    #     summary(classifier, input_size=inputs["x"])
    #     sys.stdout = sys.__stdout__  # Reset stdout to default

    # torch.save(classifier, "~/Downloads/zero_pulse_classifier.pt")
    # print(f"Model summary saved to {summary_file}")

    # summary_file = "~/Downloads/ximg_to_ypdf_autoencoder.txt"

    # # Redirect stdout to the file
    # with open(summary_file, "w") as f:
    #     sys.stdout = f
    #     summary(Ximg_to_Ypdf_Autoencoder, input_size=inputs["x"])
    #     sys.stdout = sys.__stdout__  # Reset stdout to default

    # torch.save(Ximg_to_Ypdf_Autoencoder, "~/Downloads/ximg_to_ypdf_autoencoder.pt")
    # print(f"Model summary saved to {summary_file}")
    
    

    
    
    
    
if __name__ == "__main__":
    main()