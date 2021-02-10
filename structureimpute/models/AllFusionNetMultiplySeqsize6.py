import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .utils import *

####################################################################
class AllFusionNetMultiplySeqsize6(nn.Module):
    def __init__(self, args):
        super(AllFusionNetMultiplySeqsize6, self).__init__()
        self.input_size = args.input_size
        self.use_residual = args.use_residual
        self.lstm_all_time = args.lstm_all_time
        self.lstm_output_time = args.lstm_output_time
        self.dropout = args.dropout
        self.drop = nn.Dropout(self.dropout)
        self.channels1 = args.channels1
        self.channels2 = args.channels2
        
        if self.use_residual:
            self.conv1 = nn.Sequential(
                conv(in_channels=1, out_channels=self.channels1, kernel_size=(11,self.input_size), padding=(5,0)),
                nn.BatchNorm2d(self.channels1),
                nn.ReLU(inplace=True)
            )
            
            downsample1 = nn.Sequential(                   
                nn.Conv2d(self.channels1, self.channels2, kernel_size=1, stride=1, bias=False),                    
                nn.BatchNorm2d(self.channels2)
            )


            self.resblock1 = ResidualBlock(in_channels=self.channels1, 
                                          out_channels=self.channels2, 
                                          kernel_size=(11,1), padding=(5,0), 
                                          downsample=downsample1)
            
            self.conv2 = nn.Sequential(
                conv(in_channels=1, out_channels=self.channels1, kernel_size=(11,1), padding=(5,0)),
                nn.BatchNorm2d(self.channels1),
                nn.ReLU(inplace=True)
            )
            
            downsample2 = nn.Sequential(                   
                nn.Conv2d(self.channels1, self.channels2, kernel_size=1, stride=1, bias=False),                    
                nn.BatchNorm2d(self.channels2)
            )
            
            self.resblock2 = ResidualBlock(in_channels=self.channels1, 
                                          out_channels=self.channels2, 
                                          kernel_size=(11,1), padding=(5,0), 
                                          downsample=downsample2)
        
        self.hidden_size = args.lstm_hidden_size
        self.num_layers = args.lstm_num_layers
        self.bidirectional = args.lstm_bidirectional
        self.direction = 2 if self.bidirectional else 1
        self.num_classes = args.sequence_length
        if self.use_residual:
            self.lstm = nn.LSTM(self.channels2, self.hidden_size, self.num_layers, batch_first=True, bidirectional=self.bidirectional, dropout=self.dropout) # as residual block
        else:
            self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=self.bidirectional, dropout=self.dropout)
        self.fc = nn.Linear(self.hidden_size * self.direction, self.num_classes)
        
        if self.use_residual:
            self.icSHAPE = nn.LSTM(self.channels2, self.hidden_size, self.num_layers, batch_first=True, bidirectional=self.bidirectional, dropout=self.dropout)
        else:
            self.icSHAPE = nn.LSTM(1, self.hidden_size, self.num_layers, batch_first=True, bidirectional=self.bidirectional, dropout=self.dropout)
        self.fc2 = nn.Linear(self.hidden_size * self.direction, self.num_classes)
        
    
    def forward(self, x, y):
        # print(x.shape)
        
        if self.use_residual:
            # x = self.resblock(x.unsqueeze(0).view(-1,1,100,4)).view(-1,100,4)
            x = x.view(-1,1,100,self.input_size)    # (N,1,100,4)
            y = y.view(-1,1,100,1)
            # print("[Check shape]",x.shape,y.shape)
            x = self.conv1(x)         # (N,4,100,1)
            y = self.conv2(y)
            x_conv = x
            y_conv = y * x_conv
            # print("[Check shape]",x.shape,y.shape)
            x = self.resblock1(x)      # (N,4,100,1)
            y = self.resblock2(y_conv)
            x_res = x
            y_res = y * x_res
            # print("[Check shape]",x.shape,y.shape)
            y = y_res.view(-1, self.channels2, 100).transpose(1,2)
            x = x.view(-1, self.channels2, 100).transpose(1,2) # (N,100,4)
            # print("[Check shape]",x.shape,y.shape)
        
        # Set initial hidden and cell states 
        device = x.get_device()
        h0 = torch.zeros(self.num_layers * self.direction, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * self.direction, x.size(0), self.hidden_size).to(device)
        
        h1 = torch.zeros(self.num_layers * self.direction, y.size(0), self.hidden_size).to(device)
        c1 = torch.zeros(self.num_layers * self.direction, y.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        out1, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
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
