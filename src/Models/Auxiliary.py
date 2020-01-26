import keras
import keras.layers as layer
from Layers.DownSampling import DownSampling

def Auxiliary(x1, output):
    x2 = DownSampling(256, name="downsample1", do_twice=False, x=output)(x1)
    x3 = DownSampling(128, name='downsample2')(x2)
    x4 = DownSampling(128, name='downsample3')(x3)
    x5 = DownSampling(128, name='downsample4')(x4)
    x6 = DownSampling(128, name='downsample4')(x5)
    return x1, x2, x3, x4, x5, x6