import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .utils import *

####################################################################
class AllFusionNetDE(nn.Module):
    def __init__(self, args):
        super(AllFusionNetDE, self).__init__()
        self.input_size = args.input_size
        self.use_residual = args.use_residual
        self.lstm_all_time = args.lstm_all_time
        self.lstm_output_time = args.lstm_output_time
        self.dropout = args.dropout
        self.drop = nn.Dropout(self.dropout)
        self.channels1 = args.channels1
        self.channels2 = args.channels2
        self.channels3 = args.channels3
        self.channels4 = args.channels4
        self.channels5 = args.channels5
        self.channels6 = args.channels6
        self.channels7 = args.channels7
        self.channels8 = args.channels8
        self.channels9 = args.channels9
        
        if self.use_residual:
            self.convS1 = nn.Sequential(conv(in_channels=1, out_channels=self.channels1, kernel_size=(11,4), padding=(5,0)), nn.BatchNorm2d(self.channels1), nn.ReLU(inplace=True))
            downsampleS1 = nn.Sequential(nn.Conv2d(self.channels1, self.channels2, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(self.channels2))
            self.resblockS1 = ResidualBlock(in_channels=self.channels1, out_channels=self.channels2, kernel_size=(11,1), padding=(5,0), downsample=downsampleS1)
            downsampleS2 = nn.Sequential(nn.Conv2d(self.channels2, self.channels3, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(self.channels3))
            self.resblockS2 = ResidualBlock(in_channels=self.channels2, out_channels=self.channels3, kernel_size=(11,1), padding=(5,0), downsample=downsampleS2)
            downsampleS3 = nn.Sequential(nn.Conv2d(self.channels3, self.channels4, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(self.channels4))
            self.resblockS3 = ResidualBlock(in_channels=self.channels3, out_channels=self.channels4, kernel_size=(11,1), padding=(5,0), downsample=downsampleS3)
            downsampleS4 = nn.Sequential(nn.Conv2d(self.channels4, self.channels5, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(self.channels5))
            self.resblockS4 = ResidualBlock(in_channels=self.channels4, out_channels=self.channels5, kernel_size=(11,1), padding=(5,0), downsample=downsampleS4)
            downsampleS5 = nn.Sequential(nn.Conv2d(self.channels5, self.channels6, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(self.channels6))
            self.resblockS5 = ResidualBlock(in_channels=self.channels5, out_channels=self.channels6, kernel_size=(11,1), padding=(5,0), downsample=downsampleS5)
            downsampleS6 = nn.Sequential(nn.Conv2d(self.channels6, self.channels7, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(self.channels7))
            self.resblockS6 = ResidualBlock(in_channels=self.channels6, out_channels=self.channels7, kernel_size=(11,1), padding=(5,0), downsample=downsampleS6)
            downsampleS7 = nn.Sequential(nn.Conv2d(self.channels7, self.channels8, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(self.channels8))
            self.resblockS7 = ResidualBlock(in_channels=self.channels7, out_channels=self.channels8, kernel_size=(11,1), padding=(5,0), downsample=downsampleS7)
            downsampleS8 = nn.Sequential(nn.Conv2d(self.channels8, self.channels9, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(self.channels9))
            self.resblockS8 = ResidualBlock(in_channels=self.channels8, out_channels=self.channels9, kernel_size=(11,1), padding=(5,0), downsample=downsampleS8)
            
            self.convI1 = nn.Sequential(conv(in_channels=1, out_channels=self.channels1, kernel_size=(11,1), padding=(5,0)), nn.BatchNorm2d(self.channels1), nn.ReLU(inplace=True))
            downsampleI1 = nn.Sequential(nn.Conv2d(self.channels1, self.channels2, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(self.channels2))
            self.resblockI1 = ResidualBlock(in_channels=self.channels1, out_channels=self.channels2, kernel_size=(11,1), padding=(5,0), downsample=downsampleI1)
        
        self.hidden_size = args.lstm_hidden_size
        self.num_layers = args.lstm_num_layers
        self.bidirectional = args.lstm_bidirectional
        self.direction = 2 if self.bidirectional else 1
        self.num_classes = args.sequence_length
        if self.use_residual:
            self.Seq = nn.LSTM(self.channels9, self.hidden_size, self.num_layers, batch_first=True, bidirectional=self.bidirectional, dropout=self.dropout) # as residual block
        else:
            self.Seq = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=self.bidirectional, dropout=self.dropout)
        self.fc = nn.Linear(self.hidden_size * self.direction, self.num_classes)
        
        if self.use_residual:
            self.icSHAPE = nn.LSTM(self.channels2, self.hidden_size, self.num_layers, batch_first=True, bidirectional=self.bidirectional, dropout=self.dropout)
        else:
            self.icSHAPE = nn.LSTM(1, self.hidden_size, self.num_layers, batch_first=True, bidirectional=self.bidirectional, dropout=self.dropout)
        self.fc2 = nn.Linear(self.hidden_size * self.direction, self.num_classes)
        
    
    def forward(self, x, y):
        # print(x.shape)
        
        if self.use_residual:
            x = x.view(-1,1,100,4)    
            y = y.view(-1,1,100,1)
            # print("[Check shape] view",x.shape,y.shape)
            
            x = self.convS1(x)        
            y = self.convI1(y)            
            x = self.resblockS1(x)  
            x = self.resblockS2(x)
            x = self.resblockS3(x)
            x = self.resblockS4(x)
            x = self.resblockS5(x)
            x = self.resblockS6(x)
            x = self.resblockS7(x)
            y = self.resblockI1(y)
            x = self.resblockS8(x)
           
            y = y.view(-1, self.channels2, 100).transpose(1,2)
            x = x.view(-1, self.channels9, 100).transpose(1,2) # (N,100,4)
            # print("[Check shape]",x.shape,y.shape)
        
        # Set initial hidden and cell states 
        device = x.get_device()
        h0 = torch.zeros(self.num_layers * self.direction, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * self.direction, x.size(0), self.hidden_size).to(device)
        
        h1 = torch.zeros(self.num_layers * self.direction, y.size(0), self.hidden_size).to(device)
        c1 = torch.zeros(self.num_layers * self.direction, y.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        out1, _ = self.Seq(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        # print("y shape",y.shape)
        out2, _ = self.icSHAPE(y, (h1, c1))
        # import pdb;pdb.set_trace()
        
        # 相乘的作用相当于要求更高了，比如真实的值是1，那么两个的输出都必须接近于1时loss才会小
        out = out1*out2
        # print("[Check shape] out(out1*out2): {}".format(out.shape))
        if self.lstm_all_time:
            out = self.fc(out[:, :, :])
        else:
        # Decode the hidden state of the last time step
            out = self.fc(out[:, self.lstm_output_time, :]) #   => out = self.fc(out[:, :, :])
        # print("[Check shape] out(after fc): {}".format(out.shape))
        return torch.sigmoid(out) # rescale to [0,1]
