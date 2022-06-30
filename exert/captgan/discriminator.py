import os
import torch
from torch import nn
from .dataset import CAPTCHA_LEN, IMAGE_CC, IMAGE_SIZE


class DiscriminatorNet(nn.Module):
    '''

    '''

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            # => (3, 128, 64)
            nn.Conv2d(
                IMAGE_CC,
                IMAGE_SIZE[0],
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=(1, 1),
                bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),

            # => (128, ,)
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

            # => (128 * 2, ,)
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

            # => (128 * 8, ,)
            nn.Conv2d(
                IMAGE_SIZE[0] * 8,
                1,
                kernel_size=(4, 4),
                stride=(1, 1),
                padding=0,
                bias=False
            ),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

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
