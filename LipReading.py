#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 17:45:36 2018

@author: andrey
"""
#%%
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

batchNorm1 = nn.BatchNorm2d(96)
batchNorm2 = nn.BatchNorm2d(256)
class EncoderRNN(nn.Module):

    def __init__(self):
        super(EncoderRNN, self).__init__()
        #init parameters
        self.hidden_size=256
        
        # CNN (EF)
        self.conv1 = nn.Conv2d(in_channels=5,out_channels=96,kernel_size=(3,3),stride=(1,1), padding=(1,1))
        self.conv2 = nn.Conv2d(in_channels=96,out_channels=256,kernel_size=(3,3), padding =1, stride=(2,2))
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), padding=1)
        self.conv4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3),padding=1)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3),padding=1)
        self.fc6 = nn.Linear(32768, 512) #512*8*8=32768 перейдет в 512
        
        # LSTM
        self.lstm1=nn.LSTM(512,self.hidden_size)
        self.lstm2=nn.LSTM(self.hidden_size,self.hidden_size)
        self.lstm3=nn.LSTM(self.hidden_size,self.hidden_size)
        
    def forward(self,input,hidden1,hidden2,hidden3):
        output = self.CNN(input)            
        
        output=output.view(1,1,512)
        output,hidden1=self.lstm1(output,hidden1)
        output,hidden2=self.lstm2(output,hidden2)
        output,hidden3=self.lstm3(output,hidden3)
        
        return output,hidden1,hidden2,hidden3
        
    def initHidden(self):
        return (torch.autograd.Variable(torch.randn(1, 1, self.hidden_size)),
         torch.autograd.Variable(torch.randn((1, 1, self.hidden_size))))
            
    def CNN(self, x):
         #1   
         print(x.shape)
         x = F.relu(self.conv1(x))
         print(x.shape)
         x = F.max_pool2d(x, kernel_size=(3,3),stride=2,padding=1)
         print(x.shape)
         #print("mean=",x.mean())        
         x =batchNorm1(x)
         #print("mean=",x.mean())
         print("first layer finish")
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
         x=x.view(32768)
         x = F.relu(self.fc6(x)) 
         
         
         return  x



encoder = EncoderRNN()
print(encoder)


input = Variable(torch.randn(1,5,120, 120))
hidden1 = encoder.initHidden()
hidden2 = encoder.initHidden()
hidden3 = encoder.initHidden()
out,hidden1,hidden2,hidden3 = encoder(input,hidden1,hidden2,hidden3)
print(out.shape)
#%% 