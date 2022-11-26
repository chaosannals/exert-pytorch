import sys
from loguru import logger
from exert.captgan.dataset import CaptchDataset
from exert.captgan.model2 import CaptganTrainer2

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
        'logs/captgan-train2-{time:YYYY-MM-DD}.log',
        level='INFO',
        # rotation='00:00',
        rotation='2000 KB',
        retention='7 days',
        encoding='utf8'
    )

    train_dataset = CaptchDataset('./assets/captchas')
    trainer = CaptganTrainer2('./captgan_g2.m', './captgan_d2.m')
    trainer.train(train_dataset)

if '__main__' == __name__:
    train()
