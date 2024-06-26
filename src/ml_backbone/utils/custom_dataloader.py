import numpy as np
import h5py
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
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
            
class DataMilking_Nonfat(Dataset):
    def __init__(self, root_dir = "", pulse_number = 2, transform=None, subset=None): #pulse_range is [min_pulses, max_pulses]
        self.root_dir = root_dir
        self.transform = transform
        self.shot_paths = []
        
        train_files = os.listdir(root_dir)
        train_files = list(filter(lambda x: x.endswith('.h5'), train_files))
        if subset is not None:
            train_files = train_files[:subset]
        
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
            img_x = f[shot_id]["Ximg"][()]
            img_x = torch.tensor(img_x, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
            img_y = f[shot_id]["Ypdf"][()]
            img_y = torch.tensor(img_y, dtype=torch.float32)
            if self.transform:
                img_x = self.transform(img_x)
                img_y = self.transform(img_y)
        # dataset = torch.utils.data.TensorDataset(torch.tensor(img_x), torch.tensor(img_y))    
                   
        # print("img: ", img)
        # print("attribute_data", attribute_data)
        return img_x, img_y
        # return dataset
            
transform = transforms.Compose([
    transforms.ToTensor()
])


class DataMilking_HeavyCream(Dataset):
    def __init__(self, root_dir = "", img_type="Ypdf", attributes = [], pulse_number = 2, transform=None): #pulse_range is [min_pulses, max_pulses]
        self.root_dir = root_dir
        self.transform = transform
        self.img_type = img_type
        self.attributes = attributes
        self.shot_paths = []
        self.pulse_number = pulse_number
        
        train_files = os.listdir(root_dir)
        train_files = list(filter(lambda x: x.endswith('.h5'), train_files))
        
            

    def __len__(self):
        return len(self.shot_paths)

    def __getitem__(self, idx):
        
        file_path = os.path.join(self.root_dir, self.shot_paths[idx][0]) # find file where shot is
        shot_id = self.shot_paths[idx][1] #use other index to find the shot within the file
        
        
        with h5py.File(file_path, 'r') as f:
            number_of_pulses = f[shot_id].attrs["npulses"]
            
            # One-hot encode the number of pulses using pulse_number parameter as cutoff, 0, 1, 2, 3,...,pulse_number or more
            pulse_num = torch.zeros(self.pulse_number+1)
            if number_of_pulses <= self.pulse_number:
                pulse_num[number_of_pulses] = 1
            else:
                pulse_num[self.pulse_number] = 1

            if self.img_type == "Ypdf":
                img_y = f[shot_id]["Ypdf"][()]
                img_y = torch.tensor(img_y, dtype=torch.float32)
                if self.transform:
                    img_y = self.transform(img_y)
                return img_y, pulse_num
            elif self.img_type == "Ximg":
                img_x = f[shot_id]["Ximg"][()]
                img_x = torch.tensor(img_x, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
                if self.transform:
                    img_x = self.transform(img_x)
                return img_x, pulse_num
