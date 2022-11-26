import os
from time import time_ns
from torchvision.utils import save_image
from exert.captgan.generator2 import GeneratorNet2
from exert.captgan.dataset import answer_rollv, answer_v2bt, answer_v2s, make_transform


if '__main__' == __name__:
    m = GeneratorNet2()
    m.load('./captgan_g2.m')
    tf = make_transform((34, 139))

    for i in range(2):
        cv = answer_rollv()
        ct = answer_v2bt(cv)
        cs = answer_v2s(cv)
        n = ''.join(cs)
        t = time_ns()
        p = f'ganpic/2-{t}-{n}.png'
        d = os.path.dirname(p)
        if not os.path.isdir(d):
            os.makedirs(d)
        r = m(ct) #.view(1, IMAGE_CC, 139, 34)
        save_image(r, p)
        
        print(f'generate: {p}')