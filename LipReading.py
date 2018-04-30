import sys
sys.path.append('/home/a.chernov/anaconda3/lib/python3.5/site-packages')

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from alphabet import Alphabet
class EncoderRNN(nn.Module):

    def __init__(self):
        super(EncoderRNN, self).__init__()
        # init parameters
        self.hidden_size = 256
        # CNN (EF)
        self.conv1 = nn.Conv2d(in_channels=5, out_channels=96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(3, 3), padding=1, stride=(2, 2))
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1)
        self.conv4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1)
        self.fc6 = nn.Linear(32768, 512)  # 512*8*8=32768 перейдет в 512

        # batch norm for CNN
        self.batchNorm1 = nn.BatchNorm2d(96)
        self.batchNorm2 = nn.BatchNorm2d(256)
        self.lstm1=nn.LSTM(512,self.hidden_size,num_layers=3)
#        self.lstm2=nn.LSTM(self.hidden_size,self.hidden_size)
#        self.lstm3=nn.LSTM(self.hidden_size,self.hidden_size)       
        
        
    def forward(self,input,h,c):
#        output = self.CNN(input)            
        CNN_out=Variable(torch.FloatTensor(input.shape[0],512).zero_()) # то есть первый параметр это seq_len; второй выход CNN
        
        CNN_out=self.CNN(input)
        return self.RNN(CNN_out.cuda(),h,c)

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
    def RNN(self, input,h,c):  # input.shape= seq_len*512
        input = torch.unsqueeze(input, 1).cuda()
        output, hidden = self.lstm1(input,(h,c))
        return output, hidden

#    def initHidden(self):
#        return (torch.autograd.Variable(torch.randn(1, 1, self.hidden_size).cuda()),
#                torch.autograd.Variable(torch.randn((1, 1, self.hidden_size)).cuda()))

    def CNN(self, x):
        # 1
        
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=(3, 3), stride=2, padding=1)
        x = self.batchNorm1(x)
        # 2
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=(3, 3), stride=2, padding=1)
        x = self.batchNorm2(x)

        # 3
        x = F.relu(self.conv3(x))
        
        # 4
        x = F.relu(self.conv4(x))
        
        # 5
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(x, kernel_size=(3, 3), stride=2, padding=1)
        
        # 6
        x = x.view(x.shape[0],32768)
        x = self.fc6(x)  # должна ли быть функция актвации для последнего слоя?

        return x

        
class DecoderRNN(nn.Module):
    
    def __init__(self):
        super(DecoderRNN, self).__init__()
        # LSTM
        self.hidden_size=256
        self.embedding = nn.Embedding(47, self.hidden_size)
        self.lstm1=nn.LSTMCell(self.hidden_size,self.hidden_size) 
        self.lstm2=nn.LSTMCell(self.hidden_size,self.hidden_size) 
        self.lstm3=nn.LSTMCell(self.hidden_size,self.hidden_size) 
#        attention
        self.att_fc1=nn.Linear(self.hidden_size,self.hidden_size)
        self.att_fc2=nn.Linear(self.hidden_size,self.hidden_size)
        self.att_fc3=nn.Linear(self.hidden_size,self.hidden_size)
        self.att_vector = Variable(torch.randn(1,self.hidden_size),requires_grad=True).cuda()
        self.att_W = Variable(torch.randn(self.hidden_size,self.hidden_size), requires_grad=True).cuda()
        self.att_V = Variable(torch.randn(self.hidden_size,self.hidden_size), requires_grad=True).cuda()
        self.att_b = Variable(torch.randn(self.hidden_size,1), requires_grad=True).cuda()
        
        #MLP
        self.MLP_hidden_size = 256
        self.MLP_fc1 = nn.Linear(2*self.MLP_hidden_size,self.MLP_hidden_size)        
        self.MLP_fc2 = nn.Linear(self.MLP_hidden_size,self.MLP_hidden_size)        
        self.MLP_fc3=nn.Linear(self.MLP_hidden_size,47)
        
    def forward(self,Y,h,c, outEncoder,teacher_force):# Y это кол-во символов умножить на 256
        h = h.cuda()
        c = c.cuda()
        output_decoder= torch.autograd.Variable(torch.zeros(Y.shape[0]-1, 1, 47)).cuda()
        Y = self.embedding(Y).view(Y.shape[0], 1, self.hidden_size)
        
        for  i in range(len(Y)-1): # -1 так как sos не учитывем в criterion
            if ((np.random.rand()>teacher_force)or(i==0)):
                h[0],c[0] = self.lstm1(Y[i],(h[0],c[0]))
                h[1],c[1] = self.lstm2(h[0],(h[1],c[1]))
                h[2],c[2] = self.lstm3(h[1],(h[2],c[2]))
            else:
               # print("NOT REAL TARGET")
                argmax = torch.max(output_decoder[i-1][0],dim=0)
                indMax=(argmax[1][0]).cuda()
                inputs = self.embedding(indMax).view(1, self.hidden_size)
                h[0],c[0]=self.lstm1(inputs,(h[0],c[0]))
                h[1],c[1]=self.lstm2(h[0],(h[1],c[1]))
                h[2],c[2]=self.lstm3(h[1],(h[2],c[2]))
            context = self.attention(h[2], outEncoder)
            context = torch.mm(context,outEncoder).cuda()
            output_decoder[i] = self.MLP( torch.cat( (h[2],context),1 ) )
        return output_decoder.cuda()
    
    def evaluate(self,h,c,outEncoder): # sos в return быть не должно
        h = torch.squeeze(h,0)
        c = torch.squeeze(c,0)
        max_len = 20
        result = torch.FloatTensor(max_len,1,47).zero_().cuda()
        alphabet = Alphabet()
        listArgmax=[]# буквы, которые выдал
        Y_cur = self.embedding( Variable(torch.LongTensor([alphabet.ch2index('<sos>')]).cuda()) ).view(1,self.hidden_size)
        for  i in range(max_len-1):
            j=i+1
            h[0],c[0] = self.lstm1(Y_cur,(h[0],c[0]))
            h[1],c[1] = self.lstm2(h[0],(h[1],c[1]))
            h[2],c[2] = self.lstm3(h[1],(h[2],c[2]))
            context = self.attention(h, outEncoder)
#            c = torch.mm( torch.unsqueeze(c,torch.squeeze(self.outEncoder,1) )
            context = torch.mm(context,outEncoder)
            char = self.MLP( Storch.cat( (h[2],context),1 ) )
           # print(char.data[0])
            result[j] = char.data
            argmax = torch.max(result[j][0],dim=0)
            #print(result[j][0])
            listArgmax.append(argmax[1][0])
            if argmax[1][0] == alphabet.ch2index('<eos>'):
               #print("BREAK EVAL",argmax[1][0]) 
               max_len=j+1
               break
            Y_cur=self.embedding( Variable(torch.LongTensor([argmax[1][0]]).cuda()) ).view(1,self.hidden_size)
#            print(output_decoder.shape)
        print("chars",listArgmax)
        return result[:max_len]        
 
    
    def MLP(self,v):
        v = F.relu(self.MLP_fc1(v))
        v = F.relu(self.MLP_fc2(v))
        v = self.MLP_fc3(v)
        return v 
    def attention(self,hidden, outEncoder):# то есть hidden это 1*1*256; outEncoder это 10*1*256        
        outEnSize= outEncoder.shape[0]
      #  print("hid",hidden.shape)
        hidden= torch.t(hidden.expand(outEnSize,-1))
        WS = torch.mm(self.att_W,hidden)       
#        print(outEncoder.shape)
#        print(self.att_V.shape)
        VOut = torch.mm(self.att_V,torch.t(outEncoder))
        E = F.tanh(WS + VOut + self.att_b.expand(-1,outEnSize))
        E = torch.mm(self.att_vector,E)
#        i=0
#        for out_enc_i in outEncoder:
#             out2 = torch.unsqueeze(out_enc_i,0)
#             out2=self.att_fc2(out2)
#             out=F.tanh(out1+out2)
#             out=out.view(-1,1)
#             e[i]= torch.mm(self.w,out)
#             i=i+1
        return F.softmax(E, dim=1)          

#%%        
#encoder = EncoderRNN()
#print(encoder)

#seq_len=10
#input = Variable(torch.randn(10,5,120, 120))
#hidden = encoder.initHidden()
#out,hidden = encoder(input)
#print(out.shape)
#print(hidden[0].shape)
#
#
#
#
#
#

###
#out = torch.squeeze(out,1)
#decoder= DecoderRNN()
#print(out.shape)
#count_character=100
#Y_answer=Variable(torch.torch.randn(count_character,1,47)) # верные Y
#out_RNN = decoder(Y_answer,hidden[0],hidden[1],out) #Y_NN - что выдала NN
## И в следующий декодер передаем hidden_RNN и C
#
#print(out_RNN.shape)
##print(hidden_RNN.shape)
#
##%%
## test attention
#decoder= DecoderRNN()
#hidden=Variable(torch.torch.randn(1,1,256)) 
#out=Variable(torch.torch.randn(10,1,256)) 
#decoder.attention(hidden,out)
   
#%%
#
#loss = nn.CrossEntropyLoss()
#input = Variable(torch.FloatTensor([[99, 0], [99,0]]))
#
#print(input)
#target = Variable(torch.LongTensor([0, 1]))
#output = loss(input, target)
#print(output)
#%%
