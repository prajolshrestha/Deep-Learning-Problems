import torch.nn as nn

import warnings
warnings.filterwarnings("ignore")

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        self.batch_dim = None

    def forward(self, input_tensor):

        self.batch_dim = input_tensor.shape[0]
        return input_tensor.reshape(self.batch_dim,-1)

class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride_shape= 1):
        super(ResBlock, self).__init__()
        # sequence
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size= 3,stride=stride_shape, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size= 3, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        
        # skip connection
        self.residual_conv = True
        self.conv1X1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride_shape)
        if in_channels == out_channels and stride_shape == 1:
            self.residual_conv = False
        else: 
            self.residual_conv = True
        
        # 
        self.batch_norm3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU()

        self.seq = nn.Sequential(self.conv1, self.batch_norm1, self.relu1, self.conv2, self.batch_norm2)
        self.residual = None
        self.cnt = 0

    def forward(self, input_tensor):

        # ResBlock = (conv,BN, relu, conv, BN)
        self.residual = input_tensor
        output_tensor = self.seq(input_tensor)

        # Residual layer (conv)
        if self.residual_conv:

            self.residual = self.conv1X1(self.residual)
        
        # Now normalize the residual ==> (BN, relu)
        self.residual = self.batch_norm3(self.residual)
        output_tensor += self.residual # resblock output + residual layer
        output_tensor = self.relu3(output_tensor)

        return output_tensor
    

class ResNet(nn.Module):
    def __init__(self) -> None:
        super(ResNet, self).__init__()

        self.seq1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2)
        ) 

        self.seq2 = nn.Sequential(
            ResBlock(in_channels=64, out_channels=64),
            ResBlock(in_channels=64, out_channels=128, stride_shape=2),
            ResBlock(in_channels=128, out_channels=256, stride_shape=2),
            nn.Dropout(p = 0.5),
            ResBlock(in_channels=256, out_channels=512, stride_shape=2)
        )

        self.seq3 = nn.Sequential(
            nn.AvgPool2d(kernel_size= 10),
            Flatten(),
            nn.Dropout(p = 0.5),
            nn.Linear(in_features=512, out_features=2),
            nn.Sigmoid()
        )
    
    def forward(self, input_tensor):
        output_tensor = self.seq1(input_tensor)
        output_tensor = self.seq2(output_tensor)
        output_tensor = self.seq3(output_tensor)

        return output_tensor


