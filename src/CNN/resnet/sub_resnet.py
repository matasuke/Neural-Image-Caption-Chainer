import chainer
import chainer.functions as F
import chainer.links as L
from chainer import initializers


class BottleNeckA(chainer.Chain):
    
    def __init__(self, in_size, ch, out_size, stride=2):
        initW = initializers.HeNormal()
        super(BottleNeckA, self).__init__(
                conv1 = L.Convolution2D(in_size, ch, 1, stride, 0, initialW=initW, nobias=True),
                bn1 = L.BatchNormalization(ch),
                conv2 = L.Convolution2D(ch, ch, 3, 1, 1, initialW=initW, nobias=True),
                bn2=L.BatchNormalization(ch),
                conv3=L.Convolution2D(ch, out_size, 1, 1, 0, initialW=initW, nobias=True),
                bn3=L.BatchNormalization(out_size),
                
                conv4=L.Convolution2D(in_size, out_size, 1, stride, 0, initialW=initW, nobias=True),
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
                conv1 = L.Convolution2D(in_size, ch, 1, 1, 0, initialW=initW, nobias=True),
                bn1 = L.BatchNormalization(ch),
                conv2=L.Convolution2D(ch, ch, 3, 1, 1, initialW=initW, nobias=True),
                bn2=L.BatchNormalization(ch),
                conv3=L.Convolution2D(ch, in_size, 1, 1, 0, initialW=initW, nobias=True),
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
        for i in range(layer-1):
            links += [('b{}'.format(i+1), BottleNeckB(out_size, ch))]

        for l in links:
            self.add_link(*l)
        self.forward = links

    def __call__(self, x, train):
        for name, _ in self.forward:
            f = getattr(self, name)
            x = f(x, train)

        return x

