import torch
import torch.nn as nn
import torch.nn.functional as F

class residualBox(nn.Module):

    conv1 = None
    conv2 = None
    def __init__(self, in_channels, out_channels, stride = 1, last = False):
        super().__init__(residualBox)
        self.last = last
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(out_channels))
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride),
                                          nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = nn.Sequential()
        
        def forward(self, x):
            out = self.conv1(x)
            out = self.conv2(x)
            out += self.shortcut(x)
            preact = out
            out = F.relu(out)
            if self.is_last:
                return out, preact
            else:
                return out