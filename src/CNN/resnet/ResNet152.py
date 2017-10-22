import math
import chainer
import chainer.functions as F
import chainer.links as +

class ResNet152(chainer.Chain):
    def __init__(self, n_blocks=[3, 8, 36, 3]):
        w = chainer.initializers.HeNormal()
        super(ResNet152, self).__init__(
            cov1 = L.Convolution2D(None, 64, 7, 2, 3, initialW=w, nobias=True),
            bn1 = L.BatchNormalization(64)
            )
