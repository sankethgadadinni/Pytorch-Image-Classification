import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F




class CNNNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super(CNNNet, self).__init__()

        self.conv1 = self.conv_module(in_channels, 10)

        self.conv2 = self.conv_module(10, 20)

        self.dropout = nn.Dropout()
        self.fc1 = nn.Linear(20*63*63, 1024)
        self.fc2 = nn.Linear(1024, num_classes)


    def forward(self, x):

        x = self.conv1(x)

        x = self.dropout(self.conv2(x))
    
        x = x.view(x.shape[0], -1)

        x = F.relu(self.fc1(x))

        x = F.dropout(x)
        x = self.fc2(x)

        return x


    
    def conv_module(self, in_channels, out_channels):
        
        module = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        return module
