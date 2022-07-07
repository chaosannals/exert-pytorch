import os
import torch
from torch import nn
from .dataset import CAPTCHA_LEN, IMAGE_CC, IMAGE_SIZE


class DiscriminatorNet(nn.Module):
    '''

    '''

    def __init__(self):
        super().__init__()

        si = IMAGE_CC
        c1 = int(IMAGE_SIZE[0] * IMAGE_SIZE[1] / 32)
        c2 = int(IMAGE_SIZE[0] * IMAGE_SIZE[1] / 16)
        c3 = int(IMAGE_SIZE[0] * IMAGE_SIZE[1] / 8)
        c4 = int(IMAGE_SIZE[0] * IMAGE_SIZE[1] / 4)

        self.lv1 = nn.Sequential(
            # => (3, ,)
            nn.Conv2d(
                si,
                c1,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=(1, 1),
                bias=False
            ),
            nn.BatchNorm2d(c1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.lv2 = nn.Sequential(
            # => (, , )
            nn.Conv2d(
                c1,
                c2,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=(1, 1),
                bias=False
            ),
            nn.BatchNorm2d(c2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.lv3 = nn.Sequential(
            # => (, , )
            nn.Conv2d(
                c2,
                c3,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=(1, 1),
                bias=False
            ),
            nn.BatchNorm2d(c3),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.lv4 = nn.Sequential(
            # => (, , )
            nn.Conv2d(
                c3,
                c4,
                kernel_size=(4, 4),
                stride=(3, 3),
                padding=(1, 1),
                bias=False
            ),
            nn.BatchNorm2d(c4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.out = nn.Sequential(
            # => (, ,)
            nn.Conv2d(
                c4,
                1,
                kernel_size=(3, 3),
                stride=(3, 3),
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
