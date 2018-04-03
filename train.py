from torch import nn, cuda, optim
from torch.autograd import Variable
import time
import math
import torch

from Networks.LipReading import EncoderRNN, DecoderRNN
from Networks.config import *
from Networks.data_loader import get_loader


def to_var(x, volatile=False):
    if cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def train(frames, targets, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, use_cuda):

    encoder_hidden = encoder.initHidden()

    frames = Variable(frames)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = frames.size()[0]
    target_length = targets.size()[0]
    #
    # encoder_outputs = Variable(torch.zeros(input_length, encoder.hidden_size))
    # encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0

    # for ei in range(input_length):
    #     print('Inside ei: ', frames[ei])
    #     encoder_output, encoder_hidden = encoder(frames[ei], encoder_hidden)
    #     encoder_outputs[ei] = encoder_output[0][0]

    encoder_output, encoder_hidden = encoder(frames)

    decoder_hidden = encoder_hidden

    for di in range(target_length):
        decoder_input = targets[di]
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        loss += criterion(decoder_output, targets[di])  # Тут непонятно, как учитывать <sos>?

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length


def train_iters(encoder, decoder, use_cuda, num_epochs=NUM_EPOCHS,
                print_every=10, plot_every=10, learning_rate=LEARNING_RATE):


    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Загружаем данные - в итоге получаем объект типа torch.utils.data.Dataloader,
    data_loader = get_loader()

    for epoch in range(num_epochs + 1):
        for i, (frames, targets) in enumerate(data_loader):
            frames = frames.view(-1, 5, 120, 120)
            print(frames.shape)
            # print(frames)
            loss = train(frames, targets, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, use_cuda)

            print_loss_total += loss
            plot_loss_total += loss

            # if i % print_every == 0:
            #     print_loss_avg = print_loss_total / print_every
            #     print_loss_total = 0
            #     print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
            #                                  iter, iter / n_iters * 100, print_loss_avg))


use_cuda = False
if cuda.is_available():
    print('cuda is available!')
    use_cuda = True

# Build the model
encoder = EncoderRNN()
decoder = DecoderRNN()
train_iters(encoder, decoder, use_cuda)
