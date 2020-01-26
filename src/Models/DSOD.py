import keras.backend as K
import keras.layers as layer
from Auxiliary import *
from BackBone import *
from src.Layers.PredictArm import predict_block
from Configs import CONFIG

def DSOD():
    image_size, num_classes = CONFIG['image_size'], CONFIG['num_classes']

    input = layer.Input(image_size)
    x, output = BackBone(input)
    x1, x2, x3, x4, x5, x6 = Auxiliary(x, output)

    x1_conf, x1_loc, x1_priorbox = predict_block(x1, 'x1', 3, 30.0, [2])
    x2_conf, x2_loc, x2_priorbox = predict_block(x2, 'x2', 6, 60.0, [2, 3], 114.0)
    x3_conf, x3_loc, x3_priorbox = predict_block(x3, 'x3', 6, 114.0, [2, 3], 168.0)
    x4_conf, x4_loc, x4_priorbox = predict_block(x4, 'x4', 6, 168.0, [2, 3], 222.0)
    x5_conf, x5_loc, x5_priorbox = predict_block(x5, 'x5', 6, 222.0, [2, 3], 276.0)
    x6_conf, x6_loc, x6_priorbox = predict_block(x6, 'x6', 6, 276.0, [2, 3], 330.0)

    conf = layer.Concatenate(axis=1, name='conf')([x1_conf, x2_conf, x3_conf,
                                x4_conf, x5_conf, x6_conf])
    loc = layer.Concatenate(axis=1, name='loc')([x1_loc, x2_loc, x3_loc,
                              x4_loc, x5_loc, x6_loc])
    priorbox = layer.Concatenate(axis=1, name='priorbox')([x1_priorbox, x2_priorbox, x3_priorbox,
                              x4_priorbox, x5_priorbox, x6_priorbox])

    num_boxes = K.int_shape(loc)[-1] // 4
    conf = layer.Reshape((num_boxes, num_classes), name='conf_reshape')(conf)
    conf = layer.Activation(activation='softmax', name='conf_softmax')(conf)
    loc = layer.Reshape((num_boxes, 4), name='loc_reshape')(loc)

    prediction = layer.Concatenate(axis=2, name='prediction')([loc, conf, priorbox])

    model = keras.Model(input=input, output=prediction)

    return model

model = DSOD()
model.summary()