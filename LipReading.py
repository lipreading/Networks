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
        self.lstm1=nn.LSTM(512,self.hidden_size,num_layers=1)
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
       #a hidden = self.initHidden()
        input = torch.unsqueeze(input, 1).cuda()
        # print(input.shape)
        # print(input)
        output, hidden = self.lstm1(input,(h,c))
        # print("outRNN", output.shape)
        # print("hiddenRNN", hidden[0].shape)
        return output, hidden

    def initHidden(self):
        return (torch.autograd.Variable(torch.randn(1, 1, self.hidden_size).cuda()),
                torch.autograd.Variable(torch.randn((1, 1, self.hidden_size)).cuda()))

    def CNN(self, x):
        # 1
        # print(x.shape)
        x = F.relu(self.conv1(x))
        # print(x.shape)
        x = F.max_pool2d(x, kernel_size=(3, 3), stride=2, padding=1)
        # print(x.shape)
        # print("mean=",x.mean())
        x = self.batchNorm1(x)
        # print("mean=",x.mean())
        # print("first layer finish")
        # 2
        x = F.relu(self.conv2(x))
        #  print(x.shape)
        x = F.max_pool2d(x, kernel_size=(3, 3), stride=2, padding=1)
        #  print(x.shape)
        x = self.batchNorm2(x)

        # 3
        x = F.relu(self.conv3(x))
        # print(x.shape)

        # 4
        x = F.relu(self.conv4(x))
        # print(x.shape)

        # 5
        x = F.relu(self.conv5(x))
        # print(x.shape)
        x = F.max_pool2d(x, kernel_size=(3, 3), stride=2, padding=1)
        # print(x.shape)

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
        self.lstm1=nn.LSTMCell(self.hidden_size,self.hidden_size) #кол-во букв в русском алфавите = 33 и еще + 3.
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
        self.MLP_fc3=nn.Linear(self.MLP_hidden_size,self.MLP_hidden_size)
        
    def forward(self,Y,h,c, outEncoder):# Y это кол-во символов умножить на 256
        h = torch.squeeze(h,0).cuda()
        c = torch.squeeze(c,0).cuda()
        output_decoder= torch.autograd.Variable(torch.zeros(Y.shape[0], 1, 47)).cuda()
        Y = self.embedding(Y).view(Y.shape[0], 1, self.hidden_size)
        for  i in range(len(Y)):
            h,cLSTM = self.lstm1(Y[i],(h,c))
            c = self.attention(h, outEncoder)
#            c = torch.mm( torch.unsqueeze(c,torch.squeeze(self.outEncoder,1) )
            c = torch.mm(c,outEncoder).cuda()
            output_decoder[i] = self.MLP( torch.cat( (cLSTM,c),1 ) )
#            print(output_decoder.shape)
        return output_decoder.cuda()
       # return F.log_softmax(Y,dim=1),C,hidden1,hidden2,hidden3  #         разобраться с softmax!
    
    def evaluate(self,h,c,outEncoder):
        h = torch.squeeze(h,0)
        c = torch.squeeze(c,0)
        max_len = 30
        result = Variable(torch.FloatTensor(max_len,1,47).zero_()).cuda()
        Y_cur = torch.FloatTensor(1,47).zero_().cuda()
        alphabet = Alphabet()
        Y_cur[0][alphabet.ch2index('<sos>')] = 1.0
        Y_cur=Variable(Y_cur)
        for  i in range(max_len):
            h,cLSTM = self.lstm1(Y_cur,(h,c))
            c = self.attention(h, outEncoder)
#            c = torch.mm( torch.unsqueeze(c,torch.squeeze(self.outEncoder,1) )
            c = torch.mm(c,outEncoder)
            char = self.MLP( torch.cat( (cLSTM,c),1 ) )
            result[i] = char
            Y_cur= char
            argmax = torch.max(Y_cur.data[0],dim=0)
            argmax=argmax[1]            
            if argmax[0] == alphabet.ch2index('<eos>'):
                max_len=i+1
                break
#            print(output_decoder.shape)
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
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
torch.manual_seed(1)
word_to_ix = {"hello": 0, "world": 1}
embeds = nn.Embedding(10, 6)  # 2 words in vocab, 5 dimensional embeddings
input = Variable(torch.LongTensor([[1,5,4,3,2]]))
input2 = Variable(torch.LongTensor([[1,5,4,3,2]]))

hello_embed = embeds(input)
hello_embed2 = embeds(input2)
print(hello_embed)
print(hello_embed2)
