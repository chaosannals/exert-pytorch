import os
import torch
from torch import nn
from .dataset import CAPTCHA_LEN, ANSWER_SET_LEN, IMAGE_SIZE, IMAGE_CC


class GeneratorNet(nn.Module):
    '''

    '''

    def __init__(self):
        super().__init__()

        si = int(CAPTCHA_LEN * ANSWER_SET_LEN / 2)
        c1 = int(IMAGE_SIZE[0] * IMAGE_SIZE[1] / 4)
        c2 = int(IMAGE_SIZE[0] * IMAGE_SIZE[1] / 8)
        c3 = int(IMAGE_SIZE[0] * IMAGE_SIZE[1] / 16)
        c4 = int(IMAGE_SIZE[0] * IMAGE_SIZE[1] / 32)

        self.lv1 = nn.Sequential(
            # => (50)
            nn.ConvTranspose2d(
                si,
                c1,
                kernel_size=(4, 4),
                stride=4,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(c1),
            nn.ReLU(True)
        )

        self.lv2 = nn.Sequential(
            # => (128 * 64 / 8, )
            nn.ConvTranspose2d(
                c1,
                c2,
                kernel_size=(4, 4),
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(c2),
            nn.ReLU(True)
        )

        self.lv3 = nn.Sequential(
            # => (128 * 4, )
            nn.ConvTranspose2d(
                c2,
                c3,
                kernel_size=(4, 4),
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(c3),
            nn.ReLU(True)
        )

        self.lv4 = nn.Sequential(
            # => (128 * 2, )
            nn.ConvTranspose2d(
                c3,
                c4,
                kernel_size=(4, 4),
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(c4),
            nn.ReLU(True)
        )
        self.out = nn.Sequential(
            # => (128, )
            nn.ConvTranspose2d(
                c4,
                IMAGE_CC,
                kernel_size=(4, 4),
                stride=2,
                padding=1,
                bias=False
            ),

            # => (3, 64, 128)
            nn.Tanh()
        )

    def forward(self, x):
        # print(f'in size: {x.size()}')
        x = self.lv1(x)
        # print(f'x 0 size: {x.size()}')
        x = self.lv2(x)
        # print(f'x 1 size: {x.size()}')
        x = self.lv3(x)
        # print(f'x 2 size: {x.size()}')
        x = self.lv4(x)
        # print(f'x 3 size: {x.size()}')
        x = self.out(x)
        # print(f'out size: {x.size()}')
        return x

    def load(self, path):
        '''
        '''

        if os.path.isfile(path):
            d = torch.load(path)
            self.load_state_dict(d)

    def save(self, path):
        '''
        '''

        d = self.state_dict()
        torch.save(d, path)
