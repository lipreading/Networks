
#%%
#import matplotlib.pyplot as plt
#import matplotlib.ticker as ticker
import torch
import os
from torch import cuda

from config import TRAINED_MODEL_PATH


#def show_plot(points):
#    plt.figure()
#    fig, ax = plt.subplots()
#   #  this locator puts ticks at regular intervals
#    loc = ticker.MultipleLocator(base=0.2)
#    ax.yaxis.set_major_locator(loc)
#    plt.interactive(False)
#    plt.plot(points)
#    plt.show()


def save_model(encoder, decoder,cnn):
    torch.save(encoder.state_dict(), os.path.join(TRAINED_MODEL_PATH, 'encoder'))
    torch.save(decoder.state_dict(), os.path.join(TRAINED_MODEL_PATH, 'decoder'))
    torch.save(cnn.state_dict(), os.path.join(TRAINED_MODEL_PATH, 'cnn'))

#%%
#with open('trainEpochLoss.txt') as f:
#     read_data = f.read()
#f.closed
##data=[]
##for i in range(len(read_data)):
##    data.append(read_data[i])
#print(read_data)    
##show_plot(read_data)
##%%
#numbers=[]
#with open('trainEpochLoss.txt') as input_data:
#    for each_line in input_data:
#        numbers.append(float(each_line.strip()))
##print(numbers) 
#show_plot(numbers)


def load_to_cuda(x):
    if cuda.is_available():
      # return x.cuda(device=gpus[0])
        return x.cuda()
    else:
        return x
