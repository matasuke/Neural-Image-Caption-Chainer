import math
import chainer
import chainer.functions as F
import chainer.links as L


class BottleNeckA(chainer.Chain):
    def __init__(self, in_size, ch, out_size, stride=2):
        w = math.sqrt(2)
        super(BottleNeckA, self).__init__(
            conv1=L.Convolution2D(in_size, ch, 1, stride, 0, initialW=w, nobias=True),
            bn1=L.BatchNormalization(ch),
            conv2=L.Convolution2D(ch, ch, 3, 1, 1, initialW=w, nobias=True),
            bn2=L.BatchNormalization(ch),
            conv3=L.Convolution2D(ch, out_size, 1, 1, 0, initialW=w, nobias=True),
            bn3=L.BatchNormalization(out_size),

            conv4=L.Convolution2D(in_size, out_size, 1, stride, 0, initialW=w, nobias=True),
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
        w = math.sqrt(2)
        super(BottleNeckB, self).__init__(
            conv1=L.Convolution2D(
                in_channels=in_size,
                out_channels=ch,
                ksize=1,
                stride=1,
                pad=0,
                initialW=w,
                nobias=True
            ),
            bn1=L.BatchNormalization(ch),
            conv2=L.Convolution2D(ch, ch, 3, 1, 1, initialW=w, nobias=True),
            bn2=L.BatchNormalization(ch),
            conv3=L.Convolution2D(ch, in_size, 1, 1, 0, initialW=w, nobias=True),
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

    def __call__(self, x):
        for name, _ in self.forward:
            f = getattr(self, name)
            x = f(x)

        return x


class ResNet(chainer.Chain):

    insize = 224

    def __init__(self):
        w = math.sqrt(2)
        super(ResNet, self).__init__(
            conv1=L.Convolution2D(3, 64, 7, 2, 3, initialW=w, nobias=True),
            bn1=L.BatchNormalization(64),
            res2=Block(3, 64, 64, 256, 1),
            res3=Block(4, 256, 128, 512),
            res4=Block(6, 512, 256, 1024),
            res5=Block(3, 1024, 512, 2048),
            fc=L.Linear(2048, 1000),
        )

    def clear(self):
        self.loss = None
        self.accuracy = None

    def __call__(self, x, t):
        self.clear()
        h = self.bn1(self.conv1(x))
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        h = self.res5(h)
        h = F.average_pooling_2d(h, ksize=7, stride=1)
        if t is "feature":
            return h
        h = self.fc(h)

        if chainer.config.train:
            self.loss = F.softmax_cross_entropy(h, t)
            self.accuracy = F.accuracy(h, t)

            return self.loss
        else:
            return F.softmax(h)
