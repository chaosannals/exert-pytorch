import os
from .discriminator import DiscriminatorNet
from .generator import GeneratorNet


class CaptganTrainer:
    '''

    '''

    def __init__(self, g_path, d_path):
        '''

        '''

        self.gmpath = g_path
        self.dmpath = d_path
        self.gnet = GeneratorNet()
        self.dnet = DiscriminatorNet()

        if os.path.isfile(self.gmpath):
            self.gnet.load(self.gmpath)

        if os.path.isfile(self.dmpath):
            self.dnet.load(self.dmpath)

    def train(self):
        '''

        '''
