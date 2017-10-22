import sys
import argparse
import chainer
import chainer.functions as F
from chainer import cuda
from chainer import serializers
sys.path.append('../../')
from img_proc import Img_proc
from ResNet50 import ResNet

parser = argparse.ArgumentParser()
parser.add_argument('--img', '-i', type=str, default="../../../sample_imgs/sample_img1.jpg", 
			help="image you want to predict")
parser.add_argument('--model', '-m', type=str, default="../../../data/models/cnn/ResNet50.model",
			help="model path you want to use")
parser.add_argument('--gpu', '-g', type=int, default=-1,
			help="GPU ID(put -1 if you don't use gpu)")
args = parser.parse_args()

#prepare img processer
img_proc = Img_proc("imagenet")

#prepare model
model = ResNet()
serializers.load_hdf5(args.model, model)
#model.train = False

#load image
print(args.img)
img = img_proc.load_img(args.img)

#prepare GPU settings

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()
    img = cuda.to_gpu(img, device=args.gpu)

#predict
with chainer.using_config('train', False):
    res = model(img, None)
    pred = F.softmax(res).data

if args.gpu >= 0:
    pred = cuda.to_cpu(pred)

print(pred)
