from Models.DSOD import DSOD
from Functions.Loss import MultiboxLoss
from Functions.BBox import *
from Functions.Generators import *

import pickle as pk
import keras
classes = 21
priors = pk.load(open('./priorboxes_300.ple', "rb"))
# preload = pk.load(open('./transponder.ple', "rb"))
utils = BBoxUtility(classes, priors, 0.5, 0.45)

path = "../VOC2007"

checkpoint = keras.callbacks.ModelCheckpoint('./TEMP.wt',
                                             period=5,
                                             save_weights_only=True)
callbacks = [checkpoint]

transponder = Yielder(path + "/JPEGImages/", path + "/Annotations/",
                          (300, 300, 3), 32, utils, classes=VOC2007MAP, end=4800)

confrimer = Yielder(path + "/JPEGImages/", path + "/Annotations/",
                          (300, 300, 3), 32, utils, classes=VOC2007MAP, start=4800)

model = DSOD()
optim = keras.optimizers.Adam(lr=0.001)
model.compile(optimizer=optim, loss=MultiboxLoss(classes).compute_loss)

epoch = 50

tg, cg = transponder.generate(), confrimer.generate()
history = model.fit_generator(tg, epochs=epoch, steps_per_epoch=4800 // 32, nb_val_samples=329 // 32,
                              nb_worker=1, use_multiprocessing=False, callbacks=callbacks,
                              validation_data=cg, verbose=1)