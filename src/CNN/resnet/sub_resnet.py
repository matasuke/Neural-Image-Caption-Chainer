import chainer
import chainer.functions as F
import chainer.links as L
from chainer import initializers


class BottleNeckA(chainer.Chain):
    
    def __init__(self, in_size, ch, out_size, stride=2):
        initW = initializers.HeNormal()
        super(BottleNeckA, self).__init__(
                conv1 = L.Convolution2D(in_size, ch, 1, stride, 0, initW, nobias=True),
                bn1 = L.BatchNormalization(ch),
                conv2 = L.Convolution2D(ch, ch, 3, 1, 1, initW, nobias=True),
                bn2=L.BatchNormalization(ch),
                conv3=L.Convolution2D(ch, out_size, 1, 1, 0, initW, nobias=True),
                bn3=L.BatchNormalization(out_size),
                
                conv4=L.Convolution2D(in_size, out_size, 1, stride, 0, initW, nobias=True),
                bn4=L.BatchNormalization(out_size),
            )

    def __call__(self, x):
        h1 = F.relu(self.bn1(self.conv1(x)))
        h1 = F.relu(self.bn2(self.conv2(h1)))
        h1 = self.bn3(self.conv3(h1))
        h2 = self.bn4(self.conv4(x))

        return F.relu(h1 + h2)

class BottleNeckB(chainer.Chain):

    def __init__(self, in_size, ch):
        initW = initializers.HeNormal()
        super(BottleNeckB, self).__init__(
                conv1 = L.Convolution2D(in_size, ch, 1, 1, 0, initW, nobias=True),
                bn1 = L.BatchNormalization(ch),
                conv2=L.Convolution2D(ch, ch, 3, 1, 1, initW, nobias=True),
                bn2=L.BatchNormalization(ch),
                conv3=L.Convolution2D(ch, in_size, 1, 1, 0, initW, nobias=True),
                bn3=L.BatchNormalization(in_size),
            )

    def __call__(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = self.bn3(self.conv3(h))

        return F.relu(h + x)
    

class Block(chainer.Chain):
    
    def __init__(self, layer, in_size, ch, out_size, stride=2):
        super(Block, self).__init__()
        links = [('a', BottleNeckA(in_size, ch, out_size, stride))]
        for i in range(1, layer):
            self.add_link('b{}'.format(i), BottleNeckB(out_size, ch))
        self.layer = layer

    def __call__(self, x):
        h = self.a(x)
        for i in range(1, self.layer):
            h = self['b{}'.format(i)](h)

        return h
