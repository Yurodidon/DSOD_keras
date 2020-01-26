from Configs import CONFIG
import keras.layers as layer
from src.Layers.Convolution import Conv

# w/o pooling, use to increase the number of dense blocks
# without reduce the final feature map resolution
# which consists of a 1x1 Conv, the number of channel is given in Configs.py
class WoPooling(layer.Layer):
    def __init__(self, channel, name, **kwargs):
        self.name = name
        self.channel = channel
        super(WoPooling, self).__init__(**kwargs)

    def __call__(self, inputs, **kwargs):
        conv = Conv(self.channel, (1, 1), activation='relu', name=self.name + '_conv')(inputs)
        return conv
