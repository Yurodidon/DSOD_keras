# DSOD in Keras
DSOD: Learning Deeply Supervised Object Detectors from Scratch <br>
paper link: https://arxiv.org/abs/1708.01241 <br>
### Overall
Train.py -> use this to train on VOC2007, remeber to change file path <br>
Config.py -> contain configure of DSOD, I will describe at next section <br>
src/Layers -> the layers which construct DSOD <br>
src/Models -> three part of networks of DSOD <br>
src/Utils -> stuff like generator, bounding box matching, Loss <br>
Almost everything is finished, except some details I will finish soon: <br>
1. Data Argumentation implementation <br>
2. Add more Annotations <br>
3. Training <br>
4. Better user experience <br>
5. More flexible<br>

About Training, the training of DSOD is REALLY DIFFCULT, the structure given by the paper even can't get enough memory for parameters on RTX2080Ti, but I will try my best training this and provide trained models. <br>
### About Configs
CONFIG is a dictionary, contains important hyper-parameters of DSOD, belong to Configs.py <br>
A correct CONFIG should contain: <br>
1. the size of images (image_size, tuple with length 3) <br>
2. the number of classes (num_classes, int) <br>
3. if use w/o pooling layers (useW/O, bool) <br>
4. if use stem (useStem, bool) <br>
5. number of channel in first conv layer (1stConvC, int) <br>
6. number of channel in bottleneck layer (BottleC, int) <br>
7. growth rate for dense blocks (k, int) <br>
8. compression (theta, a float belong to \[0, inf)) <br>
9. repeat time for each dense blocks (t, tuple with length 4) <br>

the CONFIG paper given is following: <br>
```
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
```

If you meet any problem, please leave a message on Issue plate or email yurodidon@gmail.com <br>
I will reply you as soon as I could~
