import os
import torch
from torch import nn, optim
from torch.utils.data.dataloader import DataLoader
from .discriminator import DiscriminatorNet
from .generator import GeneratorNet
from .dataset import answer_rollbt

LABEL_REAL = 1
LABEL_FAKE = 0


class CaptganTrainer:
    '''

    '''

    def __init__(self, g_path, d_path, learningRate=2e-4):
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

        self.criterion = nn.BCELoss()
        self.goptimizer = optim.Adam(
            self.gnet.parameters(), lr=learningRate, betas=(0.5, 0.999))
        self.doptimizer = optim.Adam(
            self.dnet.parameters(), lr=learningRate, betas=(0.5, 0.999))

    def train(self, dataset, epoch_count=100, epoch_turn=100, batch_size=128):
        '''

        '''
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4
        )

        rlabels = torch.zeros(batch_size)
        rlabels.fill_(LABEL_REAL)
        flabels = torch.zeros(batch_size)
        flabels.fill_(LABEL_FAKE)

        g_losses = 0
        d_losses = 0

        for epoch in range(epoch_count):
            if epoch > 0:
                tc = epoch * epoch_turn
                avg_glosses = g_losses / tc
                avg_dlosses = d_losses / tc
                print(f'save :{epoch} avg dl: {avg_dlosses} gl:{avg_glosses}')
                self.dnet.save(self.dmpath)
                self.gnet.save(self.gmpath)

            for i, (data, labels) in enumerate(data_loader, 0):
                if i >= epoch_turn:
                    break

                print(f'[{epoch}-{i}] start:')
                # 判别
                self.doptimizer.zero_grad()

                droutput = self.dnet(data).view(-1)
                drloss = self.criterion(droutput, rlabels)

                fakes = self.gnet(labels)
                dfoutput = self.dnet(fakes.detach()).view(-1)
                dfloss = self.criterion(dfoutput, flabels)

                dloss = drloss + dfloss
                dloss.backward()
                d_losses += dloss.item()

                drmean = droutput.mean().item()
                dfmean = dfoutput.mean().item()

                self.doptimizer.step()

                # 生成
                self.goptimizer.zero_grad()

                doutput = self.dnet(fakes).view(-1)
                gdloss = self.criterion(doutput, rlabels)
                gdloss.backward()
                g_losses += g_losses.item()

                gmean = doutput.mean().item()

                self.goptimizer.step()

                if i % 10 == 0:
                    print(f'drm: {drmean} dfm: {dfmean} gm: {gmean}')
