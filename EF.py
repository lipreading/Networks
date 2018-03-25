#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 13:41:56 2018

@author: andrey
"""
#%%
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

batchNorm1 = nn.BatchNorm2d(96)
batchNorm2 = nn.BatchNorm2d(256)
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=5,out_channels=96,kernel_size=(3,3),padding=1)
        self.conv2 = nn.Conv2d(in_channels=96,out_channels=256,kernel_size=(3,3), padding =1)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), padding=1)
        self.conv4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3),padding=1)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3),padding=1)
        self.fc6 = nn.Linear(115200, 512) #512*15*15=115200 перейдет в 512
            
    def forward(self, x):
         #1   
         x = F.relu(self.conv1(x))
         print(x.shape)
         x = F.max_pool2d(x, kernel_size=(3,3),stride=2,padding=1)
         print(x.shape)
         print("mean=",x.mean())        
         x =batchNorm1(x)
         print("mean=",x.mean())
      
         #2
         x = F.relu(self.conv2(x))
         print(x.shape)
         x = F.max_pool2d(x, kernel_size=(3,3),stride=2, padding=1) 
         print(x.shape)
         x =batchNorm2(x)
         
         
         #3
         x = F.relu(self.conv3(x))
         print(x.shape)
         
         
         #4
         x = F.relu(self.conv4(x))
         print(x.shape)
         
         
         #5
         x = F.relu(self.conv5(x))
         print(x.shape)
         x = F.max_pool2d(x, kernel_size=(3,3),stride=2, padding=1) 
         print(x.shape)
         
         
         #6
         x=x.view(115200)
         x = F.relu(self.fc6(x)) 
         
         
         return  x



net = Net()
print(net)


input = Variable(torch.randn(1,5,120, 120))
out = net(input)
print(out.shape)
print("fun")
print(out.mean()) 
#%%