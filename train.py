

#%%
import sys
sys.path.append('/home/a.chernov/anaconda3/lib/python3.5/site-packages')

from torch import nn, cuda, optim
from torch.autograd import Variable
import time
import math
import torch
from tensorboardX import SummaryWriter

from LipReading import EncoderRNN, DecoderRNN
from config import *
from data_loader import get_loader, get_loader_evaluate
from utilities import save_model, load_to_cuda
from alphabet import Alphabet
import time

writer = SummaryWriter(log_dir='tensorboard_logs')  # tensorboard logging tool

def completeNull(v,v_size,out_size):#v_size - размерность вектора, out_size - необходимая размерность
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
    
   # print("targets",targets)
    frames = frames.float()
    frames, targets = to_var(frames),to_var(targets)
    decoder_optimizer.zero_grad()
    encoder_optimizer.zero_grad()
    encoder_output, encoder_hidden = encoder(frames)
    encoder_hidden[0]= encoder_hidden[0].view(encoder_hidden[0].shape[1],encoder_hidden[0].shape[0],-1)
    encoder_hidden[1]= encoder_hidden[1].view(encoder_hidden[1].shape[1],encoder_hidden[1].shape[0],-1)
    encoder_output=encoder_output.view(encoder_output.shape[1],encoder_output.shape[0],-1)
    decoder_output = decoder(targets.view(targets.shape[1],targets.shape[0]), encoder_hidden[0],encoder_hidden[1],encoder_output,teacher_force)
    decoder_output = load_to_cuda(torch.squeeze(decoder_output,1))
    targets=targets[:,1:]# убираем sos

    loss = get_loss(load_to_cuda(torch.t(decoder_output)), load_to_cuda(targets),criterion)
   
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.data[0]


def get_loss(decoder_output,targets, criterion):
    #print('start get_loss')
    decoder_output=decoder_output.contiguous().view(decoder_output.shape[0]*decoder_output.shape[1],-1)
    targets=targets.contiguous().view(targets.shape[0]*targets.shape[1])
    return criterion(decoder_output,targets)
#    if len(targets)==len(decoder_output):
#       # print('keka')
#        return criterion(decoder_output,targets)
#    coef = abs(len(targets) - len(decoder_output)) + 1
#    if len(targets)<len(decoder_output):
#           return coef*criterion(decoder_output[:len(targets)],targets)  
#    return coef*criterion(decoder_output,targets[:len(decoder_output)])        

def train_iters(encoder, decoder, num_epochs=NUM_EPOCHS,
                print_every=10, plot_every=10, learning_rate=LEARNING_RATE):

    print('ITERATIONS: {}, BATCH SIZE: {}, LEARNING RATE: {}'
          .format(num_epochs, COUNT_FRAMES, learning_rate))
    print('====================================================================')
    start = time.time()
    plot_losses = []
    total_train_loss = 0
    plot_total_train_loss = 0

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = load_to_cuda(nn.CrossEntropyLoss(size_average=False))

    # Загружаем данные - в итоге получаем объект типа torch.utils.data.Dataloader,
    train_data_loader = get_loader(FRAME_DIR_TRAIN)
    evaluate_data_loader = get_loader_evaluate(FRAME_DIR_TEST)
    words_amount = 0
    total_test_loss=0.0
    # frames_batch
    frames_batch = []
    targets_batch = []
    batch_count = 0
    start_time=time.time()
    for epoch in range(1, num_epochs+1):
        count_words=0
        for i, (frames, targets) in enumerate(train_data_loader):
            #print('train - frames: ', frames.shape)
            # print('train - targets: ', targets.shape)

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
                teacher_force=0.0
            else:
                if epoch<=200:
                    teacher_force=0.15
                else:
                    teacher_force=0.3
            if count_words%200==0:
                print("finish words:",count_words*BATCH_SIZE)                        
            loss = train(frames, targets, encoder, decoder,  encoder_optimizer, decoder_optimizer, criterion,teacher_force)
            total_train_loss += loss
            plot_total_train_loss += loss
            count_words+=1
        
        writer.add_scalar('trainLoss', total_train_loss, epoch)

        with open('log2/totalTrain.txt', 'a') as f:
                     f.write(str(total_train_loss)+'\n')
        with open('log2/totalTest.txt', 'a') as f:
                     f.write(str(total_test_loss)+'\n')


        print('iteration: {}, loss: {}'.format(epoch, total_train_loss))
        total_train_loss = 0
        print("--- %s seconds ---" % (time.time() - start_time))
        print("count_words:",count_words*BATCH_SIZE)
        
    # TODO: для testLoss в tensorboard вставить в нужное место эту строчку:
    # writer.add_scalar('testLoss', loss, epoch)


      #  plot_losses.append(plot_total_train_loss/words_amount)
       # plot_total_train_loss = 0
        #
        # frames_batch = []  # очищаем батч
        # targets_batch = []
        # batch_count = 0

        for i, (frames, targets, is_valid) in enumerate(evaluate_data_loader):
            if not is_valid[0]:
                continue
 #           frames = torch.squeeze(frames, dim=0)  # DataLoader почему-то прибавляет лишнее измерение
#            targets = torch.squeeze(targets, dim=0)           
            test_loss = evaluate(frames,targets, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
           # print("test_loss",test_loss)
            total_test_loss+=test_loss         
            if i>100:
                break
 #   show_plot(plot_losses)

def evaluate(frames,targets, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):


    
    frames = frames.float()
    frames, targets = to_var(frames),to_var(targets)

    #print("eval frames",frames.shape)
    encoder_output, encoder_hidden = encoder(frames)
    encoder_output = torch.squeeze(encoder_output,1)

    decoder_output,word,ok = decoder.evaluate(encoder_hidden[0],encoder_hidden[1],encoder_output) 
    if (ok==False):
        return 0.0   
    decoder_output = torch.squeeze(decoder_output,1)
 #   print(targets.shape)
 #   print(decoder_output.shape)
    decoder_output=Variable(decoder_output)
    targets=targets[:,1:]# убираем sos
    res_exp=get_word(targets[0][:len(targets[0])-1].data)#записываем без eos
    seq=load_to_cuda(torch.LongTensor(decoder_output.shape[0]-1))
#    for i in range(len(seq)):
#        argmax = torch.max(decoder_output[i][0],dim=0)
#        # print(seq[i])
#        # print(argmax[1][0].data)
#        seq[i]=argmax[1][0].data[0]
#    res=get_word(seq)
    print("exp:",res_exp)
    print("res:",word)
   # print("res:",res)
    with open('log2/result.txt', 'a') as f:
        f.write("res:"+word+'\t'+"exp:"+res_exp+'\n')      
  #  loss = get_loss(load_to_cuda(decoder_output), load_to_cuda(targets),criterion)
  #  return loss.data[0]
    return -1.0

try:
#def calculate():
    if cuda.is_available():
        print('cuda is available!')

    # Build the model
  #  self.conv1=nn.DataParallel(self.conv1)
    encoder=torch.load('model8/encoder')
 #   encoder=EncoderRNN()
   # encoder =nn.DataParallel(encoder)
  #  decoder = DecoderRNN()
    decoder=torch.load('model8/decoder')
    if cuda.is_available():
        encoder.cuda()
        decoder.cuda()
#        encoder = nn.DataParallel(encoder)
        #encoder=nn.DataParallel(encoder)
#        decoder=nn.DataParallel(decoder,dim=1,device_ids=[0,1])
    train_iters(encoder, decoder)

    save_model(encoder, decoder)
    
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()
except Exception as e:  # если вдруг произошла какя-то ошибка при обучении, сохраняем модель
   print(e)
   save_model(encoder, decoder)
#calculate()
