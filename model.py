#!/usr/bin/env python3
# coding=utf-8
'''
Author: Kitiro
Date: 2021-06-10 09:48:32
LastEditTime: 2021-11-26 21:33:53
LastEditors: Kitiro
Description: 
FilePath: /web_zsl/model.py
'''
from torch import nn
from torch.nn import functional as F
import torch

# class MyModel(nn.Module):
#     def __init__(self, attr_dim, output_dim):
#         super(MyModel, self).__init__()
#         self.attr_dim = attr_dim
#         self.output_dim = output_dim
#         self.conv_out_dim = 52 * (attr_dim + 2)

#         self.conv1 = nn.Conv2d(1, 50, kernel_size=1)
#         self.bn1 = nn.BatchNorm2d(1)
#         self.conv2 = nn.Conv2d(1, 1, kernel_size=2, stride=1, padding=1)
#         self.fc = nn.Linear(self.conv_out_dim, self.output_dim)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))  #

#         x = x.view(-1, 1, 50, self.attr_dim)  # batch, channel, height , width
#         x = F.relu(self.bn1(self.conv2(x))) # 50 x feature map

#         x = x.view(-1, self.conv_out_dim)  # flatten

#         x = self.fc(x)

#         return x
class LinearModel(nn.Module):
    def __init__(self, attr_dim, output_dim):
        super(LinearModel, self).__init__()
        self.attr_dim = attr_dim
        self.output_dim = output_dim
        self.fwd = nn.Sequential(
            nn.Linear(self.attr_dim, 1024),  
            nn.ReLU(),
            nn.Linear(1024, self.output_dim),
            nn.ReLU())
            
    def forward(self, x):
        return self.fwd(x)    

class MyModel(nn.Module):
    def __init__(self, attr_dim, output_dim):
        super(MyModel, self).__init__()
        self.attr_dim = attr_dim
        self.output_dim = output_dim
        
        channel_num = 5
        self.conv1 = nn.Conv2d(1, 50, kernel_size=1)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        
        self.conv_out_dim = 50*2*attr_dim*channel_num
        self.fc = nn.Sequential(
            nn.Linear(self.conv_out_dim, 2048),  
            nn.ReLU(),
            nn.Linear(2048, self.output_dim))
        
    def forward(self, x):
        x = self.conv1(x) # 
        x = F.relu(self.conv2(x.view(-1, 1, 50, self.attr_dim)))  # batch, channel, height , width
        x = x.view(-1, self.conv_out_dim)  # flatten
        x = self.fc(x)

        return x    

class APYModel(nn.Module):
    def __init__(self, attr_dim, output_dim):
        super(APYModel, self).__init__()
        self.attr_dim = attr_dim
        self.output_dim = output_dim
        self.conv_out_dim = 48 * (attr_dim-2)

        self.conv1 = nn.Conv2d(1, 50, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(1)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1)
        self.fc = nn.Linear(self.conv_out_dim, self.output_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))  #

        x = x.view(-1, 1, 50, self.attr_dim)  # batch, channel, height , width
        x = F.relu(self.bn1(self.conv2(x))) # 50 x feature map
        #x = torch.mean(x, dim=0)  # average the feature map  
        x = x.view(-1, self.conv_out_dim)  # flatten

        x = self.fc(x)

        return x