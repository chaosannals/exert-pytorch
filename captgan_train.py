from exert.captgan.model import CaptganTrainer
from exert.captgan.dataset import CaptchDataset


def train():
    '''

    '''

    train_dataset = CaptchDataset('./assets/captchas')
    test_dataset = CaptchDataset('./assets/captchas')
    trainer = CaptganTrainer('./captgan_g.m', './captgan_d.m')
    trainer.train(train_dataset, test_dataset)


if '__main__' == __name__:
    train()
