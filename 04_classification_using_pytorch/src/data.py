from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]

# Task 1: Data Preprocessing
class ChallengeDataset(Dataset):
    
    def __init__(self, data, mode) -> None:
        """
           Arguments:
            data (string): a container structure that stores the information found in the file "data.csv".
            mode (string): either "val" or "train". 
        """
        super().__init__()

        self.data = data
        self.mode = mode

        # Normalization 
        TF = tv.transforms 
        self.val_transform = TF.Compose([TF.ToPILImage(),
                                        TF.ToTensor(),
                                        TF.Normalize(train_mean, train_std)])

        self.train_transform = TF.Compose([TF.ToPILImage(),
                                        TF.ToTensor(),
                                        TF.Normalize(train_mean, train_std)])

    
    def __len__(self):
        """
            Returns length of currently loaded data
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
            Returns sample as a tuple: image and corresponding label
        """
        if self.mode == "val":
            data = self.data.iloc[idx]
            img = imread(data['filename'], as_gray=True)
            img = gray2rgb(img)
            
            label = np.array([data['crack'], data['inactive']])
            img = self.val_transform(img)

            return img, label
        
        if self.mode == "train":
            data = self.data.iloc[idx]
            img = imread(data['filename'], as_gray=True)
            img = gray2rgb(img)
            
            label = np.array([data['crack'], data['inactive']])
            img = self.val_transform(img)

            return img, label
    
    @property
    def transform(self):
        return self._transform
    
    @transform.setter
    def transform(self, transform):
        self._transform = transform


