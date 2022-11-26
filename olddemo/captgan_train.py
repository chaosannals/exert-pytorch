import sys
from loguru import logger
from exert.captgan.model import CaptganTrainer
from exert.captgan.dataset import CaptchDataset

@logger.catch
def train():
    '''

    '''

    logger.remove()
    logger.add(
        sink=sys.stdout,
        level='INFO'
    )
    logger.add(
        'logs/captgan-train-{time:YYYY-MM-DD}.log',
        level='INFO',
        # rotation='00:00',
        rotation='2000 KB',
        retention='7 days',
        encoding='utf8'
    )

    train_dataset = CaptchDataset('./assets/captchas')
    trainer = CaptganTrainer('./captgan_g.m', './captgan_d.m')
    trainer.train(train_dataset)


if '__main__' == __name__:
    train()
