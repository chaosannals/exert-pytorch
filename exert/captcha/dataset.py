import os
import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms

ANSWER_SET = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
]


def answer_s2v(text):
    return [ANSWER_SET.index(i) for i in text]


def answer_v2s(vector):
    return [ANSWER_SET[i] for i in vector]

def make_transform(size=(128, 64)):
    return transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

class CaptchDataset(data.Dataset):
    '''
    '''

    def __init__(self, root, size=(128, 64)):
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
        a = torch.Tensor(answer_s2v(r))
        i = Image.open(p)
        d = self.transform(i.convert('RGB'))
        return d, a

    def __len__(self):
        '''
        '''

        return len(self.paths)
