import glob
import random
from PIL import Image
from exert.captcha.model import CaptchaNet
from exert.captcha.dataset import make_transform, answer_v2s

def infer(path):
    t = make_transform()
    i = Image.open(path)
    r = t(i.convert('RGB')).view(1, 3, 128, 64)
    m = CaptchaNet()
    m.load('./captcha.m')
    a = m.answer(r)
    return answer_v2s(a[0])

if __name__ == '__main__':
    ps = glob.glob('./assets/captchas/*.png')
    pmax = len(ps) - 1
    for i in range(10):
        k = random.randint(0, pmax)
        p = ps[k]
        print(p)
        a = infer(p)
        print(a)