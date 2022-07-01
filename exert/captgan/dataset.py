import os
import torch
import numpy as np
from random import randint
from PIL import Image
from torch.utils import data
from torchvision import transforms

ANSWER_SET = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ]
ANSWER_SET_LEN = len(ANSWER_SET)
ANSWER_MAX_ID = ANSWER_SET_LEN - 1

CAPTCHA_LEN = 5
IMAGE_SIZE = (64, 128)
IMAGE_CC = 3


def answer_s2v(text):
    return [ANSWER_SET.index(i) for i in text]


def answer_v2s(vector):
    return [ANSWER_SET[i] for i in vector]


def answer_v2t(vector):
    r = np.random.random((len(vector), ANSWER_SET_LEN)) * 0.1
    for i, v in enumerate(vector):
        for j in range(ANSWER_SET_LEN):
            if v == j:
                r[i][j] = 1.0
    return torch.Tensor(r)


def answer_v2bt(vector):
    r = np.random.random((len(vector), ANSWER_SET_LEN)) * 0.1
    for i, v in enumerate(vector):
        for j in range(ANSWER_SET_LEN):
            if v == j:
                r[i][j] = 1.0

    return torch.Tensor(r).view(1, int(len(vector) * ANSWER_SET_LEN / 2), 1, 2)

def answer_rollbt(batch_size):
    r = np.random.random(batch_size, CAPTCHA_LEN, ANSWER_SET_LEN) * 0.1
    for k in range(batch_size):
        for i, v in answer_rollv():
            for j in range(ANSWER_SET_LEN):
                if v == j:
                    r[k][i][j] = 1.0

    return torch.Tensor(r).view(batch_size, int(CAPTCHA_LEN * ANSWER_SET_LEN / 2), 1, 2)

def answer_rollv(length=CAPTCHA_LEN):
    return [randint(0, ANSWER_MAX_ID) for _ in range(length)]


def answer_rollt(length=CAPTCHA_LEN):
    vs = answer_rollv(length)
    r = np.random.random((length, ANSWER_SET_LEN)) * 0.2
    for i, v in enumerate(vs):
        for j in range(ANSWER_SET_LEN):
            if v == j:
                r[i][j] = 1.0
    return r


def answer_rolls(length=CAPTCHA_LEN):
    v = answer_rollv(length)
    return answer_v2s(v)


def make_transform(size=IMAGE_SIZE):
    return transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


class CaptchDataset(data.Dataset):
    '''
    '''

    def __init__(self, root, size=IMAGE_SIZE):
        '''
        '''
        self.paths = [os.path.join(root, img) for img in os.listdir(root)]
        self.transform = make_transform(size)

    def __getitem__(self, index):
        '''
        '''

        p = self.paths[index]
        s = os.path.splitext(os.path.basename(p))[0]
        r = s.split('-')[1]
        v = answer_s2v(r)
        i = Image.open(p)
        d = self.transform(i.convert('RGB'))
        return d, answer_v2t(v)

    def __len__(self):
        '''
        '''

        return len(self.paths)
