import exert.captgan.dataset  as ds
from exert.captgan.generator import GeneratorNet
from torchvision.utils import save_image

m = GeneratorNet()
m.save('./captgan_g.m')