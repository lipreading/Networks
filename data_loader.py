
#%%
import torch.utils.data as data
import os, os.path
import re
from PIL import Image
import numpy as np
import torch
import json

from config import FRAME_DIR, COUNT_FRAMES
from alphabet import Alphabet


class LipsDataset(data.Dataset):
    """Lips custom Dataset"""

    def __init__(self):
        self.frame_dir = FRAME_DIR
        self.alphabet = Alphabet()
        self.words = [name for name in os.listdir(FRAME_DIR)]
        self.count = 0
        print(FRAME_DIR)

    def __len__(self):
        return len(self.words)

    def __getitem__(self, index):

        # загружаем все кадры для слова
        curr_dir = self.frame_dir + '/' + self.words[index]
        frames_list = [name for name in os.listdir(curr_dir) if not re.match(r'__', name)]   
        if len(frames_list) < COUNT_FRAMES:
        #print(frames_list)

            is_valid = False
        else:
            is_valid = True

        frames = np.zeros((len(frames_list), 120, 120))
        count = 0
        for frame in frames_list:
            frame = np.array(Image.open(os.path.join(curr_dir, frame)).convert(mode='L').getdata()).reshape((120, 120))
            frames[count] = frame
            count += 1
        frames = torch.from_numpy(frames)

        # разбиваем на батчи
        if is_valid:
            frames = make_batches(frames)

        # загружаем субтитры
        subs_path = [name for name in os.listdir(curr_dir) if re.match(r'__', name)][0]
        with open(os.path.join(curr_dir, subs_path), 'r') as subs_file:
            subs = str(json.loads(subs_file.read())['word']).lower()
        characters = list()
        characters.append(self.alphabet.ch2index('<sos>'))
        for ch in subs:
            if self.alphabet.ch2index(ch) is None:
                is_valid = False
                break
            characters.append(self.alphabet.ch2index(ch))
        characters.append(self.alphabet.ch2index('<eos>'))

        targets = torch.LongTensor(characters)
        return frames, targets, is_valid


def get_loader():

    lips_dataset = LipsDataset()
    data_loader = torch.utils.data.DataLoader(dataset=lips_dataset,num_workers=10)
    # print(data_loader)
    return data_loader


def make_batches(data_tensor, COUNT_FRAMES=COUNT_FRAMES):
    new_size = data_tensor.shape[0] - COUNT_FRAMES + 1
    # print('new size: ', new_size)
    new_data_tensor = torch.FloatTensor(new_size, 5, 120, 120).zero_()
    # print(new_data_tensor)
    for i in range(new_size):
        new_data_tensor[i] = data_tensor[i:i+5]
    # print(new_data_tensor)
    return new_data_tensor
