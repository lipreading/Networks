#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 22:19:35 2018

@author: andrey
"""

#%%
# входные данные - это 25 черно-белых картинок размером 112*112.
# архитектура:
# применяем conv (3*3) ко всем 25 картинкам( с общими весами) c 48 фильтрами и pool 3*3;
# объединяем их (сoncat) получаем размерность: W*H*1200; (так как 48*25=1200)
# далее conv1d 1*1 c кол-во фильтров = 96 для уменьешния размерности
# потом conv2 3*3 с кол-во фильтров = 256 и pool 3*3
# потом как в VGG-M; от conv3 до fc-8

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=48,kernel_size=(3,3))
        self.conv1_2 = nn.Conv2d(in_channels=1200,out_channels=96,kernel_size=(1,1))
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256,kernel_size=(3,3))
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3))
        self.conv4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3))
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3))
        self.fc6 = nn.Linear(512, 4096) # почему в fc-6 vgg-m 6*6?
        self.fc7 = nn.Linear(4096, 4096) 
        self.fc8 = nn.Linear(4096, 500)
            
    def forward(self, x):
        print(x.shape)
        x = F.max_pool2d(F.relu(self.conv1(x)), (3, 3))
        x = x.view(1,1200,36,36)
        print(x.shape) # не уверен в этих строках.
         
        x = F.relu(self.conv1_2(x))
        print(x.shape)
        
        x = F.max_pool2d(F.relu(self.conv2(x)), (3, 3))
        print(x.shape) # [1,256,11,11]
        
        #дальше VGG-M conv3-fc8
        x = F.relu(self.conv3(x))
        print(x.shape)
        
        x = F.relu(self.conv4(x))
        print(x.shape)
       
        x = F.max_pool2d(F.relu(self.conv5(x)), (3, 3))
        print(x.shape)
        
        # fc 
        x= x.view(512)
        x = F.relu(self.fc6(x)) #скорее всего он неправильный.
        x = F.relu(self.fc7(x))
        x = self.fc8(x) 
        # получиили вектор из 500 слов; то, что ожидали
        return  F.softmax(x, dim=0)

net = Net()
print(net)


input = Variable(torch.randn(25,1, 112, 112))
out = net(input)
print(out.shape) 
print(out.sum()) #1 , потому что это распределение
