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
import numpy as np
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
        
        #batch norm for CNN
        self.batchNorm1 = nn.BatchNorm2d(96)
        self.batchNorm2 = nn.BatchNorm2d(256)    
        
        # LSTM
        self.lstm1=nn.LSTM(512,self.hidden_size)
#        self.lstm2=nn.LSTM(self.hidden_size,self.hidden_size)
#        self.lstm3=nn.LSTM(self.hidden_size,self.hidden_size)       
        
        
    def forward(self,input):
#        output = self.CNN(input)            
        CNN_out=Variable(torch.FloatTensor(input.shape[0],512).zero_()) # то есть первый параметр это seq_len; второй выход CNN
        
        for i in range(input.shape[0]):   #кидаем по одному в CNN
            CNN_in = torch.unsqueeze(input[i],0) # то есть размерность 1*5*120*120
            CNN_out[i] = self.CNN(CNN_in)
                
        print(CNN_out.shape)    # seq_len*512
        return self.RNN(CNN_out)
#        out = self.CNN(input)
#        print("finish CNN:", out.shape)
        #            CNN_out[i]=out
            
#        output=output.view(1,1,512)
#        output,hidden1=self.lstm1(output,hidden1)
#        print("output")
#        print(output.shape)
#        output,hidden2=self.lstm2(output,hidden2)
#        output,hidden3=self.lstm3(output,hidden3)
#        
#        return output,hidden
#        
    def RNN(self,input): #input.shape= seq_len*512
        hidden = self.initHidden()
        input = torch.unsqueeze(input,1)
        #print(input.shape)
        output,hidden=self.lstm1(input,hidden)
        print("outRNN",output.shape)
        print("hiddenRNN",hidden[0].shape)
        return output,hidden
    def initHidden(self):
        return (torch.autograd.Variable(torch.randn(1, 1, self.hidden_size)),
         torch.autograd.Variable(torch.randn((1, 1, self.hidden_size))))
            
    def CNN(self, x):
         #1   
        # print(x.shape)
         x = F.relu(self.conv1(x))
        # print(x.shape)
         x = F.max_pool2d(x, kernel_size=(3,3),stride=2,padding=1)
        # print(x.shape)
         #print("mean=",x.mean())        
         x =self.batchNorm1(x)
         #print("mean=",x.mean())
        # print("first layer finish")
         #2
         x = F.relu(self.conv2(x))
       #  print(x.shape)
         x = F.max_pool2d(x, kernel_size=(3,3),stride=2, padding=1) 
       #  print(x.shape)
         x =self.batchNorm2(x)
         
         
         #3
         x = F.relu(self.conv3(x))
        # print(x.shape)
         
         
         #4
         x = F.relu(self.conv4(x))
        # print(x.shape)
         
         
         #5
         x = F.relu(self.conv5(x))
        # print(x.shape)
         x = F.max_pool2d(x, kernel_size=(3,3),stride=2, padding=1) 
        # print(x.shape)
         
         
         #6
         x=x.view(32768)
         x = self.fc6(x)                  #     должна ли быть функция актвации для последнего слоя?
         
         
         return  x

# work with decoder
# вообще не особо уверен насчет входа.
# почему 4 аргумента в статье y LSTM?
# а как init с?         
class DecoderRNN(nn.Module):
    
    def __init__(self, outEncoder):
        super(DecoderRNN, self).__init__()
        # LSTM
        self.outEnSize = outEncoder.shape[0]
        self.outEncoder=outEncoder  #потом другая будет
        self.hidden_size=256
        self.lstm1=nn.LSTMCell(36,self.hidden_size) #кол-во букв в русском алфавите = 33 и еще + 3.

#        attention
        self.att_fc1=nn.Linear(self.hidden_size,self.hidden_size)
        self.att_fc2=nn.Linear(self.hidden_size,self.hidden_size)
        self.att_fc3=nn.Linear(self.hidden_size,self.hidden_size)
        self.att_vector = Variable(torch.randn(1,self.hidden_size),requires_grad=True)
        self.att_W = Variable(torch.randn(self.hidden_size,self.hidden_size), requires_grad=True)
        self.att_V = Variable(torch.randn(self.hidden_size,self.hidden_size), requires_grad=True)
        self.att_b = Variable(torch.randn(self.hidden_size,1), requires_grad=True)
        
        #MLP
        self.MLP_hidden_size = 256
        self.MLP_fc1 = nn.Linear(2*self.MLP_hidden_size,self.MLP_hidden_size)        
        self.MLP_fc2 = nn.Linear(self.MLP_hidden_size,self.MLP_hidden_size)        
        self.MLP_fc3=nn.Linear(self.MLP_hidden_size,36)
        
    def forward(self,Y,h,c):
        h = torch.squeeze(h,0)
        c = torch.squeeze(c,0)
        output_decoder= torch.autograd.Variable(torch.zeros(100, 1, 36))
        for  i in range(len(Y)):
            h,cLSTM = self.lstm1(Y[i],(h,c))
            c = self.attention(h)
#            c = torch.mm( torch.unsqueeze(c,torch.squeeze(self.outEncoder,1) )
            c = torch.mm(c,self.outEncoder)
            output_decoder[i] = self.MLP( torch.cat( (cLSTM,c),1 ) )
        return output_decoder,hidden,c
       # return F.log_softmax(Y,dim=1),C,hidden1,hidden2,hidden3  #         разобраться с softmax!
    
    def MLP(self,v):
        v = F.relu(self.MLP_fc1(v))
        v = F.relu(self.MLP_fc2(v))
        v = self.MLP_fc3(v)
        return v 
    def attention(self,hidden):# то есть hidden это 1*1*256; outEncoder это 10*1*256        
        hidden= torch.t(hidden.expand(self.outEnSize,-1))
        WS = torch.mm(self.att_W,hidden)       
        VOut = torch.mm(self.att_V,torch.t(self.outEncoder))
        E = F.tanh(WS + VOut + self.att_b.expand(-1,self.outEnSize))
        E = torch.mm(self.att_vector,E)
#        i=0
#        for out_enc_i in outEncoder:
#             out2 = torch.unsqueeze(out_enc_i,0)
#             out2=self.att_fc2(out2)
#             out=F.tanh(out1+out2)
#             out=out.view(-1,1)
#             e[i]= torch.mm(self.w,out)
#             i=i+1
        return F.softmax(E)          
        
encoder = EncoderRNN()
#print(encoder)

seq_len=10
input = Variable(torch.randn(10,5,120, 120))
hidden = encoder.initHidden()
out,hidden = encoder(input)
print(out.shape)
#print(hidden[0].shape)
#
#
#
#
#
#
print("FINISH ENCODER")
##
out = torch.squeeze(out,1)
decoder= DecoderRNN(out)

count_character=100
Y_answer=Variable(torch.torch.randn(count_character,1,36)) # верные Y
out_RNN,hidden_RNN,C = decoder(Y_answer,hidden[0],hidden[1]) #Y_NN - что выдала NN
# И в следующий декодер передаем hidden_RNN и C

print(out_RNN.shape)
#print(hidden_RNN.shape)

#%%
# test attention
decoder= DecoderRNN()
hidden=Variable(torch.torch.randn(1,1,256)) 
out=Variable(torch.torch.randn(10,1,256)) 
decoder.attention(hidden,out)

#%%