import keras.layers as layer
from src.Layers.Convolution import Conv
from keras import Model

# Transition block connect two dense blocks together
def transition(x, k0, theta, name, ret=False):
    x = layer.BatchNormalization(name=name + '_bn')(x)
    ret_conv = Conv(int(k0 * theta), (1, 1), name=name+'_conv')(x)
    # Notice: in the paper of DenseNet, use Average Pool, but DSOD uses Max Pool
    x = layer.MaxPooling2D((2, 2), strides=(2, 2), name=name+'_pool', padding='same')(ret_conv)
    # return next index of layer in order to be more convenient
    if(not ret):
        return x, int(k0 * theta)
    else:
        return x, int(k0 * theta), ret_conv

# DenseBlock class
class DenseBlock(layer.Layer):
    def __init__(self, k0, k, t, name, **kwargs):
        self.k0 = k0
        self.k = k
        self.t = t
        self.name = name

        super(DenseBlock, self).__init__(**kwargs)

    def H(self, x, l, name):
        x = layer.BatchNormalization(name=name + '_bn')(x)
        x = layer.ReLU(name=name + '_H_relu')(x)
        x = Conv(self.k0 + self.k * (l - 1), (3, 3), name=name + '_H_conv')(x)
        return x

    def __call__(self, inputs, **kwargs):
        prev = [inputs]
        for i in range(1, self.t + 1):
            if (len(prev) == 1):
                c = inputs
            else:
                c = layer.concatenate(prev, axis=-1)
            x = self.H(c, i + 1, self.name + '_' + str(i))
            prev.append(x)
        return x, self.k0 + self.k * self.t