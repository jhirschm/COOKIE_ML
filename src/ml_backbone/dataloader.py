import numpy as np
from utils import DataMilking, DataMilking_Nonfat, DataMilking_SemiSkimmed
import torch


dataset = DataMilking_SemiSkimmed(root_dir="/sdf/data/lcls/ds/prj/prjs2e21/results/even-dist_Pulses_03302024/Processed_06202024/TestMode", pulse_number=2, labels=["Ximg"])

print(dataset)

data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)


for count, (data_point, labels)  in enumerate(data_loader):
    print(count, ": data point: ", data_point)
    print(count, ": labels: ", labels)
    
    if count == 10:
        exit(1)