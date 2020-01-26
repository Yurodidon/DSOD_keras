import keras
from src.Layers.woPooling import WoPooling
from src.Layers.DenseBlock import transition, DenseBlock
from src.Layers.Stem import Stem
from Configs import CONFIG

def BackBone(input):
    x = input
    if(CONFIG['useStem']):
        x = Stem(name='stem')(input)

    k0, k, theta, t = CONFIG['image_size'][2], CONFIG['k'], CONFIG['theta'], CONFIG['t']
    dense1, k1 = DenseBlock(CONFIG['1stConvC'] * 2, k, t[0], name='dense1')(x)
    trans1, k2 = transition(dense1, k1, theta, name='trans1')

    dense2, k3 = DenseBlock(k2, k, t[1], name='dense2')(trans1)
    trans2, k4, x1 = transition(dense2, k3, theta, name='trans2', ret=True)

    dense3, k5 = DenseBlock(k4, k, t[2], name='dense3')(trans2)
    bottle1, k6 = dense3, k5
    if(CONFIG['useW/O']):
        bottle1 = WoPooling(channel=k5, name='bottle1')(dense3)

    dense4, k7 = DenseBlock(k6, k, t[3], name='dense4')(bottle1)
    bottle2, k8 = dense3, k7
    if (CONFIG['useW/O']):
        bottle2 = WoPooling(channel=k7, name='bottle2')(dense4)

    return x1, bottle2