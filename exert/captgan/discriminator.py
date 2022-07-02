import os
import torch
from torch import nn
from .dataset import CAPTCHA_LEN, IMAGE_CC, IMAGE_SIZE


class DiscriminatorNet(nn.Module):
    '''

    '''

    def __init__(self):
        super().__init__()

        # si = int(CAPTCHA_LEN * ANSWER_SET_LEN / 2)
        c1 = int(IMAGE_SIZE[0] * IMAGE_SIZE[1] / 4)
        c2 = int(IMAGE_SIZE[0] * IMAGE_SIZE[1] / 8)
        c3 = int(IMAGE_SIZE[0] * IMAGE_SIZE[1] / 16)
        c4 = int(IMAGE_SIZE[0] * IMAGE_SIZE[1] / 32)

        self.lv1 = nn.Sequential(
            # => (3, 64, 128)
            nn.Conv2d(
                IMAGE_CC,
                IMAGE_SIZE[0],
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=(1, 1),
                bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.lv2 = nn.Sequential(
            # => (64, 32, 64)
            nn.Conv2d(
                IMAGE_SIZE[0],
                IMAGE_SIZE[0] * 2,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=(1, 1),
                bias=False
            ),
            nn.BatchNorm2d(IMAGE_SIZE[0] * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.lv3 = nn.Sequential(
            # => (128, 16, 32)
            nn.Conv2d(
                IMAGE_SIZE[0] * 2,
                IMAGE_SIZE[0] * 4,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=(1, 1),
                bias=False
            ),
            nn.BatchNorm2d(IMAGE_SIZE[0] * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.lv4 = nn.Sequential(
            # => (128 * 4, , )
            nn.Conv2d(
                IMAGE_SIZE[0] * 4,
                IMAGE_SIZE[0] * 8,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=(1, 1),
                bias=False
            ),
            nn.BatchNorm2d(IMAGE_SIZE[0] * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.out = nn.Sequential(
            # => (128 * 8, ,)
            nn.Conv2d(
                IMAGE_SIZE[0] * 8,
                1,
                kernel_size=(4, 4),
                stride=5,
                padding=0,
                bias=False
            ),
            nn.Sigmoid()
        )

    def forward(self, x):
        # print(f'd in size: {x.size()}')
        x = self.lv1(x)
        # print(f'x 0 size: {x.size()}')
        x = self.lv2(x)
        # print(f'x 1 size: {x.size()}')
        x = self.lv3(x)
        # print(f'x 2 size: {x.size()}')
        x = self.lv4(x)
        # print(f'x 3 size: {x.size()}')
        x = self.out(x)
        # print(f'd out size: {x.size()}')
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
