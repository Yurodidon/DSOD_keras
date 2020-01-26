import keras.layers as layer

class Conv(layer.Layer):
    def __init__(self, channel, kernel_size, name, padding='same',
                 activation = 'relu', strides=(1, 1), **kwargs):
        self.channel = channel
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation
        self.name = name
        self.strides = strides

        super(Conv, self).__init__(**kwargs)

    def __call__(self, inputs, **kwargs):
        x = layer.BatchNormalization(name=self.name + '_BN')(inputs)
        x = layer.ReLU(name=self.name + '_relu')(x)
        x = layer.Conv2D(self.channel, self.kernel_size, name=self.name + '_conv', activation=self.activation, padding=self.padding, strides=self.strides)(x)
        return x