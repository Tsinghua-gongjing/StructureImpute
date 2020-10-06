import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

####################################################################
# convolution
def conv(in_channels, out_channels, stride=1, kernel_size=11, padding=5):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                     stride=stride, padding=padding, bias=False)

####################################################################
# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=11, padding=5, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv(in_channels, out_channels, stride=stride, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv(out_channels, out_channels, stride=stride, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        # self.downsample = nn.Sequential(conv(8, 32,kernel_size=1, stride=stride, padding=5),nn.BatchNorm2d(32))
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        # print('[Check shape] resblock', out.shape, residual.shape)
        out += residual
        out = self.relu(out)
        return out