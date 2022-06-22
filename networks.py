import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import resnet


class FCN(nn.Module):
    def __init__(self, num_input_channels=3, num_output_channels=1):
        super().__init__()
        self.resnet18 = resnet.resnet18(num_input_channels=num_input_channels)
        self.conv1 = nn.Conv2d(512, 128, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 32, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, num_output_channels, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.resnet18.features(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return self.conv3(x)

class critic_FCN(nn.Module):
    def __init__(self, num_input_channels=3, num_output_channels=1):
        super().__init__()
        self.resnet18 = resnet.resnet18(num_input_channels=num_input_channels)
        self.conv1 = nn.Conv2d(512, 128, kernel_size=4, stride=2)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 32, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=4, stride=1)
        self.conv4=nn.Conv3d(4,1,kernel_size=1,stride=1)
    def forward(self, x):

        x = self.resnet18.features(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        # x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        # x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv3(x)
        x = x.unsqueeze(0)
        return self.conv4(x).view(-1)