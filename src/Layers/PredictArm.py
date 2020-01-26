from keras.layers import *
from keras.models import Model
import keras.backend as K
from src.Layers.PriorBox import PriorBox
from src.Layers.Normalization import Normalization
from src.Layers.Convolution import Conv
from Configs import CONFIG


def predict_block(input, name, num_priors, min_size, aspect_ratio, max_size=None,
                  flip=True, clip=True, variances=[0.1, 0.1, 0.2, 0.2]):
    img_size, num_classes = CONFIG['image_size'], CONFIG['num_classes']
    input = Normalization(20, name=f'{name}_norm')(input)
    origin_loc = Conv(num_priors * 4, (3, 3), name=f"{name}_mbox_loc", padding='same')(input)
    origin_conf = Conv(num_priors * num_classes, (3, 3), name=f"{name}_mbox_conf", padding='same')(input)
    loc = Flatten(name=f"{name}_mbox_loc_flat")(origin_loc)
    conf = Flatten(name=f"{name}_mbox_conf_flat")(origin_conf)
    priorbox = PriorBox(img_size, min_size=min_size, aspect_ratio=aspect_ratio, max_size=max_size,
                        flip=flip, clip=clip, variances=variances)(input)
    return conf, loc, priorbox

