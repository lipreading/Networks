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
        while 1:
            curr_dir = self.frame_dir + '/' + self.words[index]
            frames_list = [name for name in os.listdir(curr_dir) if not re.match(r'__', name)]
            if len(frames_list) % 5 == 0:
                break
            else:
                index += 1

        # Вот тут надо определиться, что как разбивать на батчи! пока что всё, что не кратно 5, будет отсеяно,
        # вместо них некоторые кадры будут повторяться

        frames = np.zeros((len(frames_list), 120, 120))
        count = 0
        for frame in frames_list:
            frame = np.array(Image.open(os.path.join(curr_dir, frame)).convert(mode='L').getdata()).reshape((120, 120))
            # print(frame.shape)
            # print(frames.shape)
            frames[count] = frame
            count += 1
        frames = torch.from_numpy(frames)
        # print(frames.shape)
        # frames = frames.view(-1, 5, 120, 120)

        # загружаем субтитры
        subs_path = [name for name in os.listdir(curr_dir) if re.match(r'__', name)][0]
        with open(os.path.join(curr_dir, subs_path), 'r') as subs_file:
            subs = str(json.loads(subs_file.read())['word']).lower()
            # print(subs)
        characters = list()
        characters.append(self.alphabet.ch2index('<sos>'))
        for ch in subs:
            characters.append(self.alphabet.ch2index(ch))
        characters.append(self.alphabet.ch2index('<eos>'))
        # print(characters)

        # преобразовываем к виду выходной матрицы
        targets = torch.IntTensor(len(characters), 36).zero_()
        for i in range(len(characters)):
            targets[i][characters[i]] = 1
        # print(targets.shape)
        # print(frames.shape)
        return frames, targets


def collate_fn(data):
    pass


def get_loader():

    lips_dataset = LipsDataset()
    data_loader = torch.utils.data.DataLoader(dataset=lips_dataset)
    # print(data_loader)
    return data_loader


# print([name for name in os.listdir('framePack/f000005')])
# print(len([name for name in os.listdir(FRAME_DIR)]))
# words = [name for name in os.listdir(FRAME_DIR)]
# print([name for name in os.listdir(FRAME_DIR + '/' + words[0]) if not re.match(r'__', name)])
# print(re.match(r'__', '__data.json'))

# dataset = LipsDataset()
# print(dataset[800].shape)

# data_loader = get_loader()
# for i, (frames, targets) in enumerate(data_loader):
#     print(i)
#     print(frames.size())
#     # print(frames)
#     # print(targets)