
#%%
import sys
sys.path.append('/home/a.chernov/anaconda3/lib/python3.5/site-packages')

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import *
from alphabet import Alphabet
from utilities import load_to_cuda

def get_word(seq): # seq-числаf
    #print(seq)
    alphabet=Alphabet()
    s=""
    if len(seq)==0:
        return s
    for el in seq:
        #print("el:",el.data)
        if(el!=34):#хардкод
               s+=alphabet.index2ch(el)
    return s

class EncoderRNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=5, out_channels=96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(3, 3), padding=1, stride=(2, 2))
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1)
        self.conv4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1)
        self.fc6 = nn.Linear(32768, 512)  # 512*8*8=32768 перейдет в 512

        # batch norm for CNN
        self.batchNorm1 = nn.BatchNorm2d(96)
        self.batchNorm2 = nn.BatchNorm2d(256)
         
        #dropout
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        
        self.lstm1=nn.LSTM(512,256,num_layers=3,batch_first=True)   
    def forward(self,x):
        first_dim=x.shape[0]
        second_dim=x.shape[1]
       # print("x",x.shape)
       # print("f",first_dim,"s",second_dim)
        x=x.view(x.shape[0]*x.shape[1],5,120,120)      
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=(3, 3), stride=2, padding=1)
        x = self.batchNorm1(x)
        # 2
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=(3, 3), stride=2, padding=1)
        x = self.batchNorm2(x)

        # 3
        x = F.relu(self.conv3(x))
        
        x = self.dropout1(x)
        # 4
        x = F.relu(self.conv4(x))
        
        x = self.dropout2(x)
        # 5
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(x, kernel_size=(3, 3), stride=2, padding=1)
        
        # 6
        x = x.view(x.shape[0],32768)
        x = self.fc6(x)  # должна ли быть функция актвации для последнего слоя?
        x = x.view(first_dim,second_dim,512)# меняем местами first и second, так как в lstm первая это seq_len, вторая - batch
        h = load_to_cuda(Variable(torch.zeros(3,x.shape[0],256)))
        c = load_to_cuda(Variable(torch.zeros(3,x.shape[0],256)))

        self.lstm1.flatten_parameters()
        output, hidden = self.lstm1(x,(h,c))
        #self.lstm1.flatten_parameters()
        hidden=list(hidden)
        hidden[0]=hidden[0].view(hidden[0].shape[1],hidden[0].shape[0],-1)
        hidden[1]=hidden[1].view(hidden[1].shape[1],hidden[1].shape[0],-1)
        return output,hidden

        
class DecoderRNN(nn.Module):
    
    def __init__(self):
        super(DecoderRNN, self).__init__()
        # LSTM
        self.hidden_size=256
        self.embedding = nn.Embedding(48, self.hidden_size)
        self.lstm1=nn.LSTMCell(self.hidden_size,self.hidden_size) 
        self.lstm2=nn.LSTMCell(self.hidden_size,self.hidden_size) 
        self.lstm3=nn.LSTMCell(self.hidden_size,self.hidden_size) 
#        attention
        self.att_fc1=nn.Linear(self.hidden_size,self.hidden_size)
        self.att_fc2=nn.Linear(self.hidden_size,self.hidden_size)
        self.att_fc3=nn.Linear(self.hidden_size,self.hidden_size)
        self.att_vector = load_to_cuda(Variable(torch.randn(1,self.hidden_size),requires_grad=True))
        self.att_W = load_to_cuda(Variable(torch.randn(self.hidden_size,self.hidden_size), requires_grad=True))
        self.att_V = load_to_cuda(Variable(torch.randn(self.hidden_size,self.hidden_size), requires_grad=True))
        self.att_b = load_to_cuda(Variable(torch.randn(self.hidden_size,1), requires_grad=True))
        
        #MLP
        self.MLP_hidden_size=256
        self.MLP_fc1 = nn.Linear(2*self.MLP_hidden_size,self.MLP_hidden_size)        
        self.MLP_fc2 = nn.Linear(self.MLP_hidden_size,self.MLP_hidden_size)        
        self.MLP_fc3=nn.Linear(self.MLP_hidden_size,48)
        
    def forward(self,Y,h,c, outEncoder,teacher_force):# Y это кол-во символов умножить на 256
        print("Y",Y)
        if (np.random.rand()>teacher_force):
            seq_len=Y.shape[0]-1
            output_decoder= load_to_cuda(torch.autograd.Variable(torch.zeros(seq_len, h.shape[1], 48)))
            Y = self.embedding(Y)
            for  i in range(len(Y)-1): # -1 так как sos не учитывем в criterion
                print("clone")
                h[0],c[0] = self.lstm1(Y[i],(h[0].clone(),c[0].clone()))
                h[1],c[1] = self.lstm2(h[0].clone(),(h[1].clone(),c[1].clone()))
                h[2],c[2] = self.lstm3(h[1].clone(),(h[2].clone(),c[2].clone()))
                context = self.attention(h[2].clone(), outEncoder)
                context = load_to_cuda( torch.bmm( context,outEncoder.view(outEncoder.shape[1],outEncoder.shape[0],-1) ) )
                print("context",context) # torch sueeze
                cat =torch.cat( (h[2].clone(),context) ,1 )
                output_decoder[i] = self.MLP(cat)    
        else:
            seq_len=Y.shape[0]-1
            output_decoder= load_to_cuda(torch.autograd.Variable(torch.zeros(seq_len, h.shape[1], 48)))
            alphabet = Alphabet()
            Y_cur = self.embedding( load_to_cuda(Variable(torch.LongTensor([alphabet.ch2index('<sos>')]))) ).view(1,self.hidden_size)
            for  i in range(seq_len-1):
                Y_cur=Y_cur.expand(BATCH_SIZE,self.hidden_size)
                h[0],c[0] = self.lstm1(Y_cur,(h[0].clone(),c[0].clone()))
                h[1],c[1] = self.lstm2(h[0].clone(),(h[1].clone(),c[1].clone()))
                h[2],c[2] = self.lstm3(h[1].clone(),(h[2].clone(),c[2].clone()))
                context = self.attention(h[2].clone(), outEncoder)
                context = load_to_cuda( torch.bmm( context,outEncoder.view(outEncoder.shape[1],outEncoder.shape[0],-1) ) )
                char = self.MLP( torch.cat( (h[2].clone(),context),1 ) )
                output_decoder[i] = char.clone()
                print("output decoder",output_decoder.shape)
                argmax = torch.max(output_decoder[i][0],dim=0)
                Y_cur=self.embedding( Variable(load_to_cuda(torch.LongTensor([argmax[1][0].data[0]]))) ).view(1,self.hidden_size)
        return output_decoder[:seq_len] 
        
        
    def evaluate(self,h0,c0,outEncoder): # sos в return быть не должно
        h = load_to_cuda(torch.squeeze(h0.clone(),0))
        c = load_to_cuda(torch.squeeze(c0.clone(),0))
        seq_len = 50# максимальная длина
        result = load_to_cuda(torch.FloatTensor(seq_len,1,48).zero_())
        alphabet = Alphabet()
        listArgmax=[]# буквы, которые выдал
        Y_cur = self.embedding( Variable(load_to_cuda(torch.LongTensor([alphabet.ch2index('<sos>')]))) ).view(1,self.hidden_size)
        for  i in range(seq_len-1):
            h[0],c[0] = self.lstm1(Y_cur,(h[0],c[0]))
            h[1],c[1] = self.lstm2(h[0],(h[1],c[1]))
            h[2],c[2] = self.lstm3(h[1],(h[2],c[2]))
            context = self.attention(h[2], outEncoder)
            context = torch.mm(context,outEncoder)
            char = self.MLP( torch.cat( (h[2],context),1 ) )
            result[i] = char.data
            argmax = torch.max(result[i][0],dim=0)
            listArgmax.append(argmax[1][0])
            if argmax[1][0] == alphabet.ch2index('<eos>'):
               seq_len=i+1
               break
            Y_cur=self.embedding( Variable(load_to_cuda(torch.LongTensor([argmax[1][0]]))) ).view(1,self.hidden_size)

        word=get_word(torch.LongTensor(listArgmax))
        print("res:",word)
        with open('log2/result.txt', 'a') as f:
                 f.write("res:"+word+'\n')
        return result[:seq_len]        
 
    
    def MLP(self,v):
        v = F.relu(self.MLP_fc1(v))
        v = F.relu(self.MLP_fc2(v))
        v = self.MLP_fc3(v)
        return v 
    def attention(self,hidden, outEncoder):# то есть hidden это 1*1*256; outEncoder это 10*1*256, если batch_size=1        
        outEnSize= outEncoder.shape[0]
        hidden= hidden.expand(outEnSize,-1,-1)
        hidden=hidden.contiguous().view(hidden.shape[1],self.hidden_size,hidden.shape[0])
        WS = torch.bmm(self.att_W.expand(BATCH_SIZE,-1,-1),hidden)
        VOut = torch.bmm(self.att_V.expand(BATCH_SIZE,-1,-1),outEncoder.view(BATCH_SIZE,self.hidden_size,outEnSize))
        E = F.tanh(WS + VOut + self.att_b.expand(BATCH_SIZE,-1,outEnSize))
        E = torch.bmm(self.att_vector.expand(BATCH_SIZE,-1,-1),E)
        print("E",E.shape)
        return F.softmax(E, dim=2)          

