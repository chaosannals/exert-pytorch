import os
import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms

ANSWER_SET = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
]

def answer_s2v(text):
    return [ANSWER_SET.index(i) for i in text]

def answer_v2s(vector):
    return [ANSWER_SET[i] for i in vector]

class CaptchDataset(data.Dataset):
    '''
    '''

    def __init__(self, root, size=(32,32)):
        '''
        '''
        self.paths = [os.path.join(root, img) for img in os.listdir(root)]
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5, 0.5, 0.5])
        ])

    def __getitem__(self, index):
        '''
        '''

        p = self.paths[index]
        s = os.path.splitext(os.path.basename(p))[0]
        a = torch.Tensor(answer_s2v(s))
        i = Image.open(p)
        d = self.transform(i)
        return d, a

    def __len__(self):
        '''
        '''

        return len(self.paths)
