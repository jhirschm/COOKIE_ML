import numpy as np
from utils import DataMilking
import torch


dataset = DataMilking(root_dir="/sdf/data/lcls/ds/prj/prjs2e21/results/even-dist_Pulses_03302024/Processed_06202024/TestMode", attributes=["energies", "phases", "npulses"], pulse_number=2)

print(dataset)

data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)


for count, batch in enumerate(data_loader):
    print("count: ", batch)
    if count == 10:
        exit(1)