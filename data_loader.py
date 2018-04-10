import torch.utils.data as data
import os, os.path
import re
from PIL import Image
import numpy as np
import torch
import json

from Networks.config import FRAME_DIR, BATCH_SIZE
from Networks.alphabet import Alphabet


class LipsDataset(data.Dataset):
    """Lips custom Dataset"""

    def __init__(self):
        self.frame_dir = FRAME_DIR
        self.alphabet = Alphabet()
        self.words = [name for name in os.listdir(FRAME_DIR)]
        self.count = 0

    def __len__(self):
        return len(self.words)

    def __getitem__(self, index):

        # загружаем все кадры для слова
        curr_dir = self.frame_dir + '/' + self.words[index]
        frames_list = [name for name in os.listdir(curr_dir) if not re.match(r'__', name)]

        frames = np.zeros((len(frames_list), 120, 120))
        count = 0
        for frame in frames_list:
            frame = np.array(Image.open(os.path.join(curr_dir, frame)).convert(mode='L').getdata()).reshape((120, 120))
            frames[count] = frame
            count += 1
        frames = torch.from_numpy(frames)

        # разбиваем на батчи
        frames = make_batches(frames)

        # загружаем субтитры
        subs_path = [name for name in os.listdir(curr_dir) if re.match(r'__', name)][0]
        with open(os.path.join(curr_dir, subs_path), 'r') as subs_file:
            subs = str(json.loads(subs_file.read())['word']).lower()
        characters = list()
        characters.append(self.alphabet.ch2index('<sos>'))
        for ch in subs:
            characters.append(self.alphabet.ch2index(ch))
        characters.append(self.alphabet.ch2index('<eos>'))

        targets = torch.LongTensor(characters)
        return frames, targets


def get_loader():

    lips_dataset = LipsDataset()
    data_loader = torch.utils.data.DataLoader(dataset=lips_dataset)
    # print(data_loader)
    return data_loader


def make_batches(data_tensor, batch_size=BATCH_SIZE):
    new_size = data_tensor.shape[0] - batch_size + 1
    new_data_tensor = torch.FloatTensor(new_size, 5, 120, 120).zero_()
    # print(new_data_tensor)
    for i in range(new_size):
        new_data_tensor[i] = data_tensor[i:i+5]
    # print(new_data_tensor)
    return new_data_tensor
