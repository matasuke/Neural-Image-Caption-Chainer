import os
import chainer
from chainer import functions as F
from chainer import links as L


class VGG16(chainer.Chain):
    def __init__(self):
        super(VGG16, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(3, 64, 3, stride=1, pad=1),
            self.conv1_2 = L.Convolution2D(64, 64, 3, stride=1, pad=1),
            
