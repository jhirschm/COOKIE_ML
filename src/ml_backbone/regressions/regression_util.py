import os
import argparse
import sys
import time
import json

# # External library imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import h5py
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import copy
import joblib

from sklearn.metrics import mean_squared_error

import datetime
from typing import List, Any
from torchinfo import summary

def get_conv_output_size(input_size, conv_layers):
    x = torch.randn(input_size)
    model = nn.Sequential(*[layer for layer_pair in conv_layers for layer in layer_pair if layer is not None])
    x = model(x)
    return x.shape
