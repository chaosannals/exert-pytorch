import torch
import torch.nn as nn
from torch.nn.modules import padding

class Bottleneck131(nn.Module):
    expansion = 4

    def __init__(self, in_planes, out_planes, stride=1):
        super().__init__()
        self.lv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_planes,
                out_channels=out_planes,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(out_planes)
        )
        self.lv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_planes,
                out_channels=out_planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_planes)
        )
        r_planes = self.expansion * out_planes
        self.lv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_planes,
                out_channels=r_planes,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(r_planes)
        )
        self.sc = nn.Sequential()
        if stride != 1 or in_planes != r_planes:
            self.sc = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_planes,
                    out_channels=r_planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(r_planes)
            )
    def forward(self, x):
        x = torch.relu(self.lv1(x))
        x = torch.relu(self.lv2(x))
        x = self.lv3(x)
        x += self.sc(x)
        return torch.relu(x)
        