

#%%
import sys
# sys.path.append('/home/a.chernov/anaconda3/lib/python3.5/site-packages')

from torch import nn, cuda, optim
from torch.autograd import Variable
import time
import math
import torch

from LipReading import EncoderRNN, DecoderRNN
from config import *
from data_loader import get_loader
from utilities import save_model


def to_var(x, volatile=False):
    if cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def train(frames, targets_for_training, targets, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, use_cuda):

    encoder_hidden = encoder.initHidden()

    frames = frames.float()
    targets_for_training = targets_for_training.float()
    frames, targets_for_training, targets = Variable(frames), \
                                            Variable(targets_for_training), Variable(targets)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # input_length = frames.size()[0]
    # target_length = targets_for_training.size()[0]
    print("frames",frames.shape)
    encoder_output, encoder_hidden = encoder(frames)
    encoder_output = torch.squeeze(encoder_output,1)
    decoder_output = decoder(targets_for_training, encoder_hidden[0],encoder_hidden[1],encoder_output)
    
    decoder_output = torch.squeeze(decoder_output,1)
#    print(targets.shape)
#    targets = torch.squeeze(targets,1)
    loss = criterion(decoder_output, targets)

    # print(loss.data[0])
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0]


def train_iters(encoder, decoder, use_cuda, num_epochs=NUM_EPOCHS,
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
    criterion = nn.CrossEntropyLoss()

    # Загружаем данные - в итоге получаем объект типа torch.utils.data.Dataloader,
    data_loader = get_loader()
    words_amount = 0

    for epoch in range(1, num_epochs+1):
        for i, (frames, targets, is_valid) in enumerate(data_loader):
            # print(frames)
            # print(is_valid[0])
            if not is_valid[0]:
                continue
            frames = torch.squeeze(frames, dim=0)  # DataLoader почему-то прибавляет лишнее измерение
            targets = torch.squeeze(targets, dim=0)
            # print(frames.shape)

            targets_for_training = torch.LongTensor(targets.shape[0], 47).zero_()
            for j in range(targets.shape[0]):
                targets_for_training[j][targets[j]] = 1
            # print('targets for training: ', targets_for_training.shape)
            targets_for_training = targets_for_training.view(-1, 1, 47)
            # print('targets: ', targets.shape)
            # print('frames: ', frames.shape)
            if i%10==0:
                test_loss = evaluate(frames, targets_for_training, targets, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, use_cuda)
                print("test_loss",test_loss)
                with open('short/testLossAdam.txt', 'a') as f:
               	     s = 'epoch=' + str(epoch) +' i=' + str(i) + 'test_loss=' +str(test_loss)+'\n'
                     f.write(s)            
            loss = train(frames, targets_for_training, targets, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, use_cuda)
            print("finished words:",i+1)
            print("train_loss",loss)
            with open('short/trainLossAdam.txt', 'a') as f:
                     s = 'epoch=' + str(epoch) +' i=' + str(i) + 'train_loss=' +str(loss)+'\n'                     
                     f.write(s) 
            print_loss_total += loss
            plot_loss_total += loss
            words_amount += 1
        
        with open('short/trainLossAdam.txt', 'a') as f:
                     s = 'finish epoch=' + str(epoch) + 'train_loss=' +str(print_loss_total)+'\n'
                     f.write(s)
        print('iteration: {}, loss: {}'.format(epoch, print_loss_total/words_amount))
        print_loss_total = 0

        plot_losses.append(plot_loss_total/words_amount)
        plot_loss_total = 0

 #   show_plot(plot_losses)

def evaluate(frames, targets_for_training, targets, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, use_cuda):

    encoder_hidden = encoder.initHidden()

    frames = frames.float()
    targets_for_training = targets_for_training.float()
    frames, targets_for_training, targets = Variable(frames), \
                                            Variable(targets_for_training), Variable(targets)


    encoder_output, encoder_hidden = encoder(frames)
    encoder_output = torch.squeeze(encoder_output,1)

    decoder_output = decoder.evaluate(encoder_hidden[0],encoder_hidden[1],encoder_output)    
    decoder_output = torch.squeeze(decoder_output,1)
 #   print(targets.shape)
 #   print(decoder_output.shape)	
    if len(targets)<=len(decoder_output):
        loss = criterion(decoder_output[:len(targets)], targets) 
        return loss.data[0]
    if len(targets)>len(decoder_output):
        loss = criterion(decoder_output, targets[:len(decoder_output)])
        return loss.data[0]   

use_cuda = False
if cuda.is_available():
    print('cuda is available!')
    use_cuda = True

# Build the model
encoder = EncoderRNN()
decoder = DecoderRNN()
train_iters(encoder, decoder, use_cuda)

save_model(encoder, decoder)
