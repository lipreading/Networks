

#%%
import sys
sys.path.append('/home/a.chernov/anaconda3/lib/python3.5/site-packages')

from torch import nn, cuda, optim
from torch.autograd import Variable
import time
import math
import torch

from LipReading import EncoderRNN, DecoderRNN
from config import *
from data_loader import get_loader
from utilities import save_model, load_to_cuda
from alphabet import Alphabet


def completeNull(v,v_size,out_size):#v_size - размерность вектора, out_size - необходимая размерность
    alphabet=Alphabet()
    newv = Variable(load_to_cuda(torch.LongTensor(out_size).zero_()))
    for i in range(v_size):
         newv[i]=v[i].clone()
 #   j=v_size
  #  while j<out_size:
  #      newv[j]=alphabet.ch2index('null')
  #      j+=1
    return newv    

def to_var(x):
    if cuda.is_available():
        x = x.cuda()
    return Variable(x)

def get_word(seq): # seq-числа
    #print(seq)
    alphabet=Alphabet()
    s=""
    if len(seq)==0:
        return s
    for el in seq:
        #print("el:",el.data)
        s+=alphabet.index2ch(el)
    return s
def train(frames, targets, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,teacher_force):

    #encoder_hidden = encoder.initHidden()

    frames = frames.float()
    frames, targets = to_var(frames),to_var(targets)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # input_length = frames.size()[0]
    # target_length = targets_for_training.size()[0]
    #print("frames",frames.shape)
    h0 = load_to_cuda(Variable(torch.zeros(3, BATCH_SIZE, encoder.hidden_size)))
    c0 = load_to_cuda(Variable(torch.zeros(3, BATCH_SIZE, encoder.hidden_size)))
       
    encoder_output, encoder_hidden = encoder(frames,h0,c0)
    encoder_output = torch.squeeze(encoder_output,1)
    #print(encoder_hidden[0].shape)
    decoder_output = decoder(targets, encoder_hidden[0],encoder_hidden[1],encoder_output,teacher_force)
    
    decoder_output = load_to_cuda(torch.squeeze(decoder_output,1))
#    print(targets.shape)
#    targets = torch.squeeze(targets,1)
    # print(targets)
    targets=targets[1:]# убираем sos
   # print("Len",len(targets),len(decoder_output))
    loss = get_loss(load_to_cuda(decoder_output), load_to_cuda(targets),criterion)
   
    # print(loss.data[0])
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0]

#def get_loss(decoder_output,targets, criterion):
#    if len(targets)==len(decoder_output):
#        return criterion(decoder_output,targets)
#    else:
#        if len(targets)<len(decoder_output):
#           compTarg =completeNull(targets,len(targets),len(decoder_output)).clone()
#           return criterion(decoder_output,compTarg)
#        else:
#            assert False,"len target > len decoder_output"

def get_loss(decoder_output,targets, criterion):
    if len(targets)==len(decoder_output):
        return criterion(decoder_output,targets)
    coef = abs(len(targets) - len(decoder_output)) + 1
    if len(targets)<len(decoder_output):
           return coef*criterion(decoder_output[:len(targets)],targets)  
    return coef*criterion(decoder_output,targets[:len(decoder_output)])        

def train_iters(encoder, decoder, num_epochs=NUM_EPOCHS,
                print_every=10, plot_every=10, learning_rate=LEARNING_RATE):

    print('ITERATIONS: {}, BATCH SIZE: {}, LEARNING RATE: {}'
          .format(num_epochs, COUNT_FRAMES, learning_rate))
    print('====================================================================')
    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = load_to_cuda(nn.CrossEntropyLoss())

    # Загружаем данные - в итоге получаем объект типа torch.utils.data.Dataloader,
    train_data_loader = get_loader(FRAME_DIR_TRAIN)
    evaluate_data_loader = get_loader(FRAME_DIR_TEST)
    words_amount = 0
    total_test_loss=0.0
    # frames_batch
    frames_batch = []
    targets_batch = []
    batch_count = 0
    for epoch in range(1, num_epochs+1):
        for i, (frames, targets) in enumerate(train_data_loader):
            print('train - frames: ', frames.shape)
            print('train - targets: ', targets.shape)

            # print(frames)
            # print(is_valid[0])

            # frames = torch.squeeze(frames, dim=0)  # DataLoader почему-то прибавляет лишнее измерение
            # targets = torch.squeeze(targets, dim=0)

            # print(frames.shape)

            #targets_for_training = torch.LongTensor(targets.shape[0], 48).zero_()
            #for j in range(targets.shape[0]):
            #    targets_for_training[j][targets[j]] = 1
            # print('targets for training: ', targets_for_training.shape)
            #targets_for_training = targets_for_training.view(-1, 1, 48)
            # print('targets: ', targets.shape)
            # print('frames: ', frames.shape)
            if epoch<=300:
                teacher_force=0.5
            else:
                if epoch<=200:
                    teacher_force=0.15
                else:
                    teacher_force=0.3                        
            loss = train(frames, targets, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,teacher_force)
           # print("finished words:",i+1)
           #print("train_loss",loss)
           # print("tot",total_test_loss)
            with open('log2/trainLoss.txt', 'a') as f:
                     s = 'epoch=' + str(epoch) +' i=' + str(i) + 'train_loss=' +str(loss)+'\n'                     
                     f.write(s) 
            print_loss_total += loss
            plot_loss_total += loss
            words_amount += 1
        
        with open('log2/trainLoss.txt', 'a') as f:
                     s = 'finish epoch=' + str(epoch) + 'train_loss=' +str(print_loss_total)+'\n'
                     f.write(s)
        with open('log2/totalTrain.txt', 'a') as f:
                     f.write(str(print_loss_total)+'\n')
        with open('log2/totalTest.txt', 'a') as f:
                     f.write(str(total_test_loss)+'\n')


        print('iteration: {}, loss: {}'.format(epoch, print_loss_total))
        print_loss_total = 0

        plot_losses.append(plot_loss_total/words_amount)
        plot_loss_total = 0
        #
        # frames_batch = []  # очищаем батч
        # targets_batch = []
        # batch_count = 0

#        for i, (frames, targets, is_valid) in enumerate(evaluate_data_loader):
#            if not is_valid[0]:
#                continue
#            frames = torch.squeeze(frames, dim=0)  # DataLoader почему-то прибавляет лишнее измерение
#            targets = torch.squeeze(targets, dim=0)           
#            test_loss = evaluate(frames,targets, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
#            print("test_loss",test_loss)
#            total_test_loss+=test_loss  
#            with open('log2/testLoss.txt', 'a') as f:
#                s = 'epoch=' + str(epoch) +' i=' + str(i) + 'test_loss=' +str(test_loss)+'\n'
#                f.write(s)            

 #   show_plot(plot_losses)

def evaluate(frames,targets, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):

   # encoder_hidden = encoder.initHidden()
    h0 = load_to_cuda(Variable(torch.zeros(3, 1, encoder.hidden_size)))
    c0 = load_to_cuda(Variable(torch.zeros(3, 1, encoder.hidden_size)))

    
    frames = frames.float()
    frames, targets = to_var(frames),to_var(targets)


    encoder_output, encoder_hidden = encoder(frames,h0,c0)
    encoder_output = torch.squeeze(encoder_output,1)

    decoder_output = decoder.evaluate(encoder_hidden[0],encoder_hidden[1],encoder_output)    
    decoder_output = torch.squeeze(decoder_output,1)
 #   print(targets.shape)
 #   print(decoder_output.shape)
    decoder_output=Variable(decoder_output)
    targets=targets[1:]# убираем sos
    res_exp=get_word(targets[:len(targets)-1].data)#записываем без eos
    seq=load_to_cuda(torch.LongTensor(decoder_output.shape[0]-1))
#    for i in range(len(seq)):
#        argmax = torch.max(decoder_output[i][0],dim=0)
#        # print(seq[i])
#        # print(argmax[1][0].data)
#        seq[i]=argmax[1][0].data[0]
#    res=get_word(seq)
    print("exp:",res_exp)
   # print("res:",res)
    with open('log2/result.txt', 'a') as f:
        f.write("exp:"+res_exp+'\n')      
    loss = get_loss(load_to_cuda(decoder_output), load_to_cuda(targets),criterion)
    return loss.data[0]


try:
    if cuda.is_available():
        print('cuda is available!')

    # Build the model
    encoder = EncoderRNN()
    decoder = DecoderRNN()
    if cuda.is_available():
        encoder.cuda()
        decoder.cuda()
    train_iters(encoder, decoder)

    save_model(encoder, decoder)
except Exception as e:  # если вдруг произошла какя-то ошибка при обучении, сохраняем модель
   print(e)
   # save_model(encoder, decoder)

