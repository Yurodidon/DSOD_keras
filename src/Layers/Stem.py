from Configs import CONFIG
import keras.layers as layer
from src.Layers.Convolution import Conv
import keras

# The Stem block with 3 conv and 1 pooling
'''
3×3 conv, stride 2
3×3 conv, stride 1
3×3 conv, stride 1
2×2 max pool, stride 2
'''
class Stem(layer.Layer):
    def __init__(self, name, **kwargs):
        self.name = name
        super(Stem, self).__init__(**kwargs)

    def __call__(self, inputs, **kwargs):
        conv1 = Conv(CONFIG['1stConvC'], (3, 3), strides=(2, 2), activation='relu', name=self.name + '_conv1', padding='same')(inputs)
        conv2 = Conv(CONFIG['1stConvC'], (3, 3), activation='relu', name=self.name + '_conv2', padding='same')(conv1)
        conv3 = Conv(CONFIG['1stConvC'] * 2, (3, 3), activation='relu', name=self.name + '_conv3', padding='same')(conv2)
        pool = layer.MaxPooling2D((2, 2), strides=(2, 2), name=self.name + '_pool')(conv3)
        return pool
