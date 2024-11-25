from lstm_pulseNum_classifier import CustomLSTMClassifier

import torch
import os
import sys
import numpy as np
import torch
import torch.nn as nn



groq_path = "/home/jhirschm/groqflow"
sys.path.append(groq_path) # Assumes on rdsrv420 with groqflow installed correctly
from groqflow import groqit

def main():
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Input Data Paths and Output Save Paths

    
    data = {
        "hidden_size": 128,#128
        "num_lstm_layers": 3,
        "bidirectional": True,
        "fc_layers": [32, 64],
        "dropout": 0.2,
        "lstm_dropout": 0.2,
        "layerNorm": False,
        # Other parameters are default or not provided in the example
    }   

    # Assuming input_size and num_classes are defined elsewhere
    input_size = 512  # Define your input size
    num_classes = 5   # Example number of classes

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

    batch_size = 1
    height, width = 16, 512
    inputs = {"x": torch.randn(batch_size, height, width)}

    gmodel1 = groqit(classModel, inputs, groqview=True)

    estimate = gmodel1.estimate_performance()
    print("Your Classifier build's estimated performance is:")
    print(f"{estimate.latency:.7f} {estimate.latency_units}")
    print(f"{estimate.throughput:.1f} {estimate.throughput_units}")

    

   

    
   
    
    
if __name__ == "__main__":
    main()