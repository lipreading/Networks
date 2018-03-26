#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 13:41:56 2018

@author: andrey
"""
# %%
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from PIL import Image


batchNorm1 = nn.BatchNorm2d(96)
batchNorm2 = nn.BatchNorm2d(256)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=5, out_channels=96, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1)
        self.conv4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1)
        self.fc6 = nn.Linear(115200, 512)  # 512*15*15=115200 перейдет в 512

    def forward(self, x):
        # 1   
        x = F.relu(self.conv1(x))
        print(x.shape)
        x = F.max_pool2d(x, kernel_size=(3, 3), stride=2, padding=1)
        print(x.shape)
        print("mean=", x.mean())
        x = batchNorm1(x)
        print("mean=", x.mean())

        # 2
        x = F.relu(self.conv2(x))
        print(x.shape)
        x = F.max_pool2d(x, kernel_size=(3, 3), stride=2, padding=1)
        print(x.shape)
        x = batchNorm2(x)

        # 3
        x = F.relu(self.conv3(x))
        print(x.shape)

        # 4
        x = F.relu(self.conv4(x))
        print(x.shape)

        # 5
        x = F.relu(self.conv5(x))
        print(x.shape)
        x = F.max_pool2d(x, kernel_size=(3, 3), stride=2, padding=1)
        print(x.shape)

        # 6
        x = x.view(115200)
        x = F.relu(self.fc6(x))

        return x


net = Net()
# print(net)

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# optimizer = optim.Adam(net.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()


def data_load():
    frames = np.zeros((5, 120, 120))
    boundary = (100, 100, 220, 220)
    for i in range(1, 6):
        img = np.array(Image.open('frames/{0}.jpg'.format(1)).convert(mode='L').crop(boundary)
                       .getdata()).reshape((120, 120))
        frames[i - 1] = img
    return torch.from_numpy(frames).view(1, 5, 120, 120)


def train(epoch, train_data, target_data):

    for _ in range(epoch):

        running_loss = 0.0
        train_data, target_data = Variable(train_data), Variable(target_data)
        optimizer.zero_grad()  # обнуляем градиенты

        # forward + backward + optimize
        output_data = net(train_data)
        loss = criterion(output_data, target_data)  # считаем потери ??
        # здесь что-то с размерностью output_data не сходится: должно быть (N, C), где C - количество классов

        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]


# train_data = data_load()
train_data = torch.randn(1, 5, 120, 120)
target_data = torch.IntTensor(512).zero_()
target_data[0] = 5

# input = Variable(torch.randn(1, 5, 120, 120))
# out = net(input)
# print(out.shape)
# print("fun")
# print(out.mean())
# %%

train(2, train_data, target_data)

input = Variable(torch.randn(1, 5, 120, 120))
output = net(input)
print(output.shape)
print(output)

