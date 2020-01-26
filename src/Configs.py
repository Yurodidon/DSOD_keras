CONFIG = {
    'image_size' : (300, 300, 3),
    'num_classes' : 21,
    'useW/O' : True, # if use transition w/o pooling
    'useStem' : True, # if use stem
    '1stConvC' : 64, # number of channels in 1st conv layer
    'BottleC' : 192, # number of channels in bottleneck layer(1 x 1 convolution)
    'k' : 48,        # growth rate in DenseBlock
    'theta' : 1,     # compression factor in transition layers.
    't' : [6, 8, 8, 8] # repeat time in dense blocks
}