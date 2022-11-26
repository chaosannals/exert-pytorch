import exert.captgan.dataset  as ds
from exert.captgan.discriminator import DiscriminatorNet
from torchvision.utils import save_image

if '__main__' == __name__:
    m = DiscriminatorNet()
    m.load('./captgan_d.m')

    