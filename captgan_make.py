import os
import exert.captgan.dataset  as ds
from exert.captgan.generator import GeneratorNet
from torchvision.utils import save_image

G_PATH = './captgan_g.m'
D_PATH = './captgan_d.m'

if not os.path.isfile(G_PATH):
    m = GeneratorNet()
    m.save('./captgan_g.m')

if not os.path.isfile(D_PATH):
    m = GeneratorNet()
    m.save('./captgan_d.m')