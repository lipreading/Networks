from torch import nn, cuda, optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence

from Networks.LipReading import EncoderRNN, DecoderRNN
from Networks.config import *
from Networks.data_loader import get_loader


def to_var(x, volatile=False):
    if cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def train():

    data_loader = get_loader()

    # Build the model
    encoder = EncoderRNN()
    decoder = DecoderRNN()

    # Load to GPU
    if cuda.is_available():
        encoder.cuda()
        decoder.cuda()

    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.parameters())
    optimizer = optim.Adam(params, lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        for i, (frames, subs, lengths) in enumerate(data_loader):

            frames = to_var(frames, volatile=True)
            targets = pack_padded_sequence(subs, length, batch_first=True)[0]

            decoder.zero_grad()
            encoder.zero_grad()
            features, hidden = encoder(frames)
            outputs, hiddenRNN = decoder(features, hidden)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()


