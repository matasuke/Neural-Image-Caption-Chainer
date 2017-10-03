import chainer
import chainer.functions as F
import chainer.links as L
from chainer import initializers
from sub_resnet import Block

class ResNet(chainer.Chain):

    def __init__(self):
        initW = initialzers.HeNormal()
        super(ResNet, self).__init__(
                conv1 = L.Convolution2D(3, 64, 7, 2, 3, initW, nobias=True),
                bn1 = L.BatchNormalization(64),
                self.res2 = Block(3, 64, 64, 256, 1),
                self.res3 = Block(4, 256, 128, 512),
                self.res4 = Block(6, 512, 256, 1024),
                self.res5 = Block(3, 1024, 512, 2048),
                self.fc = L.Linear(2048, 1000),
        )
        self.train = True
        
    def clear():
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
        h = F.average_pooling_2d(h, 7, stride=1)
        if t=="feature":
            return h
        h = self.fc(h)

        return h
