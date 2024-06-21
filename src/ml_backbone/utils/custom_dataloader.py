import numpy as np
import h5py
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch
class DataMilking(Dataset):
    def __init__(self, root_dir = "", img_type="Ypdf", attributes = [], pulse_number = 2, transform=None): #pulse_range is [min_pulses, max_pulses]
        self.root_dir = root_dir
        self.transform = transform
        self.img_type = img_type
        self.attributes = attributes
        self.shot_paths = []
        
        train_files = os.listdir(root_dir)
        train_files = list(filter(lambda x: x.endswith('.h5'), train_files))
        
        for file in train_files:
            # print("file: ", file)
            with h5py.File(os.path.join(root_dir, file), 'r') as f:
                for shot in f.keys():
                    # print("shot: ", shot)
                    # print("attributes: ", list(f[shot].attrs.items()))
                    
                    if pulse_number == f[shot].attrs["npulses"]:
                        self.shot_paths.append([file, shot])
            

    def __len__(self):
        return len(self.shot_paths)

    def __getitem__(self, idx):
        
        file_path = os.path.join(self.root_dir, self.shot_paths[idx][0]) # find file where shot is
        shot_id = self.shot_paths[idx][1] #use other index to find the shot within the file
        
        with h5py.File(file_path, 'r') as f:
            
            img  = f[shot_id][self.img_type][()]
            if self.transform:
                img = self.transform(img)
            
            attribute_data = []
            for attribute in self.attributes:
                attribute_data.append(f[shot_id].attrs[attribute])
        
        # print("img: ", img)
        # print("attribute_data", attribute_data)
        return img, attribute_data
            

transform = transforms.Compose([
    transforms.ToTensor()
])


