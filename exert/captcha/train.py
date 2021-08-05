from model import CaptchaTrainer
from dataset import CaptchDataset

def train():
    '''
    '''

    train_dataset = CaptchDataset('assets/captcha')


    trainer = CaptchaTrainer('captcha.m')
    trainer.train(train_dataset, test_dataset)
    


if __name__ == '__main__':
    train()
