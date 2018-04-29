#import matplotlib.pyplot as plt
#import matplotlib.ticker as ticker
import torch
import os

from config import TRAINED_MODEL_PATH


#def show_plot(points):
#    plt.figure()
#    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
#    loc = ticker.MultipleLocator(base=0.2)
#    ax.yaxis.set_major_locator(loc)
#    plt.interactive(False)
#    plt.plot(points)
#    plt.show()


def save_model(encoder, decoder):
    torch.save(encoder.state_dict(), os.path.join(TRAINED_MODEL_PATH, 'encoder'))
    torch.save(decoder.state_dict(), os.path.join(TRAINED_MODEL_PATH, 'decoder'))
