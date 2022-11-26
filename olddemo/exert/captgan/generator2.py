import os
import torch
from torch import nn
from .dataset import CAPTCHA_LEN, ANSWER_SET_LEN, IMAGE_SIZE, IMAGE_CC


class GeneratorNet2(nn.Module):

    def __init__(self):
        '''

        '''

        super().__init__()

        self.hs = CAPTCHA_LEN
        si = int(CAPTCHA_LEN * ANSWER_SET_LEN / 2)
        c1 = int(IMAGE_SIZE[0] * IMAGE_SIZE[1])
        c2 = int(IMAGE_SIZE[0] * IMAGE_SIZE[1] / 2)
        c3 = int(IMAGE_SIZE[0] * IMAGE_SIZE[1] / 4)
        c4 = int(IMAGE_SIZE[0] * IMAGE_SIZE[1] / 8)
        c5 = int(IMAGE_SIZE[0] * IMAGE_SIZE[1] / 16)
        c6 = int(IMAGE_SIZE[0] * IMAGE_SIZE[1] / 32)

        # print(f'si: {si}  c1: {c1}  c2: {c2}  c3: {c3}  c4: {c4}  c5:{c5}  c6: {c6}')

        self.c2 = c2
        self.c3 = c3
        self.c4 = c4

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

        self.lv2rnn = nn.LSTM(
            input_size=c1,
            hidden_size=c2,
            num_layers=self.hs,
            batch_first=True,
        )
        self.lv2 = nn.Sequential(
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

        self.lv4rnn = nn.LSTM(
            input_size=c3,
            hidden_size=c4,
            num_layers=self.hs,
            batch_first=True,
        )
        self.lv4 = nn.Sequential(
            nn.BatchNorm2d(c4),
            nn.ReLU(True)
        )

        self.lv5 = nn.Sequential(
            nn.ConvTranspose2d(
                c4,
                c5,
                kernel_size=(4, 4),
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(c5),
            nn.ReLU(True)
        )

        self.lv6 = nn.Sequential(
            nn.ConvTranspose2d(
                c5,
                c6,
                kernel_size=(4, 4),
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(c6),
            nn.ReLU(True)
        )

        self.out = nn.Sequential(
            # => (128, )
            nn.ConvTranspose2d(
                c6,
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
        '''

        '''

        # print(f'g in size: {x.size()}')

        x1 = self.lv1(x)
        s1 = x1.size()
        x1 = x1.reshape(s1[0], -1, s1[1])
        # print(f'x 1 size: {s1} => {x1.size()}')

        x2, h2 = self.lv2rnn(x1, None)
        s2 = x2.size()
        x2 = x2.reshape(s2[0], s2[2], int(s2[1] / 16), -1)
        # print(f'x 2 size: {s2} => {x2.size()}')
        x2 = self.lv2(x2)

        x3 = self.lv3(x2)
        s3 = x3.size()
        x3 = x3.reshape(s3[0], -1, s3[1])
        # print(f'x 3 size: {s3} => {x3.size()}')

        x4, h4 = self.lv4rnn(x3, None)
        s4 = x4.size()
        x4 = x4.reshape(s4[0], s4[2], int(s4[1] / 16), -1)
        # print(f'x 4 size: {s4} => {x4.size()}')
        x4 = self.lv4(x4)

        x5 = self.lv5(x4)
        # print(f'x 5 size: {x5.size()}')
        x6 = self.lv6(x5)
        # print(f'x 6 size: {x6.size()}')

        x = self.out(x6)
        # print(f'g out size: {x.size()}')
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
