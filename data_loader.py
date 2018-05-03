
#%%
import torch.utils.data as data
import os, os.path
import re
from PIL import Image
import numpy as np
import torch
import json

from config import *
from alphabet import Alphabet


class LipsDataset(data.Dataset):
    """Lips custom Dataset"""

    def __init__(self, frame_dir):
        self.frame_dir = frame_dir
        self.alphabet = Alphabet()
        # self.words = [name for name in os.listdir(FRAME_DIR)]

        # для сквозного прохода по папкам с видео
        self.words = []
        for root, dirs, files in os.walk(self.frame_dir):
            if not dirs:
                self.words.append(root)

            # print('root: ', root)
            # print('dirs: ', dirs)
            # print('files: ', files)
        print(self.words)
        self.count = 0

    def __len__(self):
        return len(self.words)

    def __getitem__(self, index):

        # загружаем все кадры для слова
        curr_dir = self.words[index]
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
        #print('get_item - targets: ', targets)
        return frames, targets, is_valid


def collate_fn(data):
    frames, targets, is_valid = zip(*data)

    # print('collate_fn - raw targets: ', targets)
    #print('collate_fn - raw frames shape: ', frames[0].shape)

    targets_lengths = [len(target) for target in targets]
    # print('collate_fn - targets_lengths: ', targets_lengths)
    batch_targets = torch.zeros(len(targets), max(targets_lengths)).long()
    for i, target in enumerate(targets):
        end = targets_lengths[i]
        batch_targets[i, :end] = target[:end]
    # print('collate_fn - batch_targets: ', batch_targets)

    frames_lengths = [frame.shape[0] for frame in frames]
  #  print('collate_fn - frames_lengths: ', frames_lengths)
    batch_frames = torch.zeros(len(frames), max(frames_lengths), COUNT_FRAMES, 120, 120).long()
    for i, frame in enumerate(frames):
        end = frames_lengths[i]
        batch_frames[i, :end] = frame[:end]
  #  print('collate_fn - batch_frames: ', batch_frames.shape)

    return batch_frames, batch_targets  # batch_targets.shape = BATCH_SIZE*max_targets_length
                                        # batch_frames.shape = BATCH_SIZE*max_frames_length*5*120*120


def get_loader(frame_dir):

    lips_dataset = LipsDataset(frame_dir)
    data_loader = torch.utils.data.DataLoader(dataset=lips_dataset, num_workers=20,
                                              collate_fn=collate_fn, batch_size=BATCH_SIZE)
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


