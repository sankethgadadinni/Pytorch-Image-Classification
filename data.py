import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader



class ImageDataset(Dataset):

    def __init__(self, base_dir, path, transform=None, train=True):
        super().__init__()

        self.base_dir = base_dir
        self.transform = transform
        self.path = path
        self.data = pd.read_csv(path)
        self.train = train

    
    def __len__(self):
        return len(self.data)

    
    def __getitem__(self, index):

        image_paths = list(self.data.iloc[:,0])

        if self.train:
            labels = list(self.data.iloc[:,1])
            label = labels[index]

        image_path = image_paths[index]
        
        full_image_path = os.path.join(self.base_dir, image_path)

        image = cv2.imread(full_image_path)


        if self.transform is not None:
            image = self.transform(image)
        
        if self.train:
            return image, label
        
        else:
            return image



