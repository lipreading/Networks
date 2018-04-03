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


 # Load to GPU
if cuda.is_available():
    print('cuda is available!')
    use_cuda = True


def train(frames, targets, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):

    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = frames.size()[0]

    encoder_outputs = Variable(torch.zeros(input_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(frames[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]

    dec


def trainIters(encoder, decoder, num_epochs=NUM_EPOCHS, print_every=10, plot_every=10, learning_rate=LEARNING_RATE):
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
            frames.view(-1, 5, 120, 120)
            loss = train(frames, targets, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

            print_loss_total += loss
            plot_loss_total += loss

            # if i % print_every == 0:
            #     print_loss_avg = print_loss_total / print_every
            #     print_loss_total = 0
            #     print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
            #                                  iter, iter / n_iters * 100, print_loss_avg))



# Build the model
encoder = EncoderRNN()
decoder = DecoderRNN()