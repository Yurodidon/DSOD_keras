import Configs
import keras.layers as layer
from src.Layers.Convolution import Conv
import keras

CONFIG = Configs.CONFIG
# down-sampling block
# consists of a 2×2, stride = 2 max pooling layer followed by a 1×1, stride = 1 conv-layer
# and a 1x1x256 conv followed by a 3x3x256 strides 2 conbined together
# for the first one, only have 2×2, stride = 2 max pooling layer followed by a 1×1, stride = 1 conv-layer
# for further infomation, please refer to [1]
# [1]: www.arxiv.org/abs/1708.01241

class DownSampling(layer.Layer):
    def __init__(self, channel, name, do_twice=True, x=None, **kwargs):
        self.channel = channel
        self.do_twice = do_twice
        self.x = x
        self.name = name
        super(DownSampling, self).__init__(**kwargs)

    def __call__(self, inputs, **kwargs):
        pool = layer.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name=self.name + '_pool')(inputs)
        conv1 = Conv(CONFIG['BottleC'], (1, 1), activation='relu', padding='same', name=self.name + '_conv1')(pool)
        if(not self.do_twice):
            return layer.concatenate([conv1, self.x], axis=-1)

        conv2 = Conv(self.channel, (1, 1), strides=(1, 1), activation='relu', padding='same', name=self.name + '_conv2')(inputs)
        conv3 = Conv(self.channel, (3, 3), strides=(2, 2), activation='relu', padding='same', name=self.name + '_conv3')(inputs)

        output = layer.concatenate([conv1, conv3], axis=-1)
        return output


