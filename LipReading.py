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
    
    def __init__(self, out_encoder):
        super(DecoderRNN, self).__init__()
        # LSTM
        self.output_encoder_size=10  #потом другая будет
        self.output_encoder=out_encoder  #потом другая будет
        self.hidden_size=256
        self.lstm1=nn.LSTM(36,self.hidden_size) #кол-во букв в русском алфавите = 33 и еще + 3.
#        self.lstm2=nn.LSTM(self.hidden_size,self.hidden_size)
#        self.lstm3=nn.LSTM(self.hidden_size,self.hidden_size)

#        attention
        self.att_fc1=nn.Linear(self.hidden_size,self.hidden_size)
        self.att_fc2=nn.Linear(self.hidden_size,self.hidden_size)
        self.att_fc3=nn.Linear(self.hidden_size,self.hidden_size)
        self.w = Variable(torch.randn(1,256))
        
        #MLP
        self.MLP_hidden_size = 256
        self.fc1 = nn.Linear(256,self.MLP_hidden_size)        
        self.fc2=nn.Linear(self.MLP_hidden_size,36)
    def forward(self,Y,hidden):
        
#        output = F.relu(output)  в статье его еще пропустили через relu
#        output,hidden=self.lstm1(Y,hidden)
#        
#        print(output.shape)
#        print(hidden.shape)
#        
#        output = F.relu(output) # так было в статье про машинный перевод
        #hidden= torch.unsqueeze(hidden,0) # то есть размерность 1*5*120*120           
        output, hidden = self.lstm1(Y, hidden)
        print("output after LSTM:",output.shape)   
        Y = output
        c = self.attention(hidden[0],torch.unsqueeze(self.output_encoder,1)) 
        c = torch.mm( torch.unsqueeze(c,0),torch.squeeze(self.output_encoder,1) )
    
        print(c.shape)
        output=Variable(torch.FloatTensor(Y.shape[0],36).zero_())  # 36- размер алфавита
        #        
        for i in range(len(Y)): # что идет в MLP? мне кажется неправильно.
            yi=torch.unsqueeze(Y[i],0)
            yi = F.relu(self.fc1(yi))    
            yi = self.fc2(yi)
            yi = self.fc3(yi)
            output[i]=yi        
        
        
        print(output.shape)
        return F.log_softmax(output, dim=1),hidden,c
       # return F.log_softmax(Y,dim=1),C,hidden1,hidden2,hidden3  #         разобраться с softmax!
    
    def attention(self,hidden,outEncoder):# то есть hidden это 1*1*256; outEncoder это 10*1*256
        print(outEncoder.shape)
        out1 = self.att_fc1(hidden)
        e=Variable(torch.FloatTensor(outEncoder.shape[0]).zero_())
        i=0
        for out_enc_i in outEncoder:
             out2 = torch.unsqueeze(out_enc_i,0)
             out2=self.att_fc2(out2)
             out=F.tanh(out1+out2)
             out=out.view(-1,1)
             e[i]= torch.mm(self.w,out)
             i=i+1
#        return F.softmax(e)     
        return e     
        
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
decoder= DecoderRNN(out)

count_character=100
Y_answer=Variable(torch.torch.randn(count_character,1,36)) # верные Y
out_RNN,hidden_RNN,C = decoder(Y_answer,hidden) #Y_NN - что выдала NN
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