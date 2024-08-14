import torch
import torch.nn as nn

conv_layers = [[nn.Conv2d(1, 16, kernel_size=3, padding=2), nn.ReLU()]]
model = nn.Sequential(*[layer for layer_pair in conv_layers for layer in layer_pair if layer is not None])

x = torch.randn(1, 1, 28, 28)
x = model(x)

print(x.shape)  # Should print the output shape