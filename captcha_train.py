from exert.captcha.model import CaptchaTrainer
from exert.captcha.dataset import CaptchDataset

def train():
    '''
    '''

    train_dataset = CaptchDataset('./assets/captchas')
    test_dataset = CaptchDataset('./assets/captchas')
    trainer = CaptchaTrainer('./captcha.m')
    trainer.train(train_dataset, test_dataset)

if __name__ == '__main__':
    train()
