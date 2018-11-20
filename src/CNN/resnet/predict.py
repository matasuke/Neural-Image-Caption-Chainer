from pathlib import Path
import argparse
import numpy as np
import chainer
import chainer.functions as F
from chainer import cuda, serializers

from src.img_proc import Img_proc
from .ResNet50 import ResNet


SAMPLE_IMG_PATH = Path('sample_imgs/COCO_test2014_000000160008.jpg')
MODEL_PATH = Path('data/models/cnn/ResNet50.model')
LABEL_PATH = Path('src/CNN/resnet/synset_words.txt')

parser = argparse.ArgumentParser()
parser.add_argument(
    '--img', '-i', type=str,
    default=SAMPLE_IMG_PATH.as_posix(),
    help="image you want to predict"
)
parser.add_argument(
    '--model', '-m', type=str,
    default=MODEL_PATH.as_posix(),
    help="model path you want to use"
)
parser.add_argument(
    '--gpu', '-g', type=int, default=-1,
    help="GPU ID(put -1 if you don't use gpu)"
)
args = parser.parse_args()

# prepare img processer
img_proc = Img_proc("imagenet")

# prepare model
model = ResNet()
serializers.load_hdf5(args.model, model)

# load image
print(args.img)
img = img_proc.load_img(args.img)

# prepare GPU settings

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()
    img = cuda.to_gpu(img, device=args.gpu)

# predict
with chainer.using_config('train', False):
    pred = model(img, None).data

if args.gpu >= 0:
    pred = cuda.to_cpu(pred)

# results
with LABEL_PATH.open() as f:
    synsets = f.read().split('\n')[:-1]

for i in np.argsort(pred)[0][-1::-1][:5]:
    print(synsets[i], pred[0][1])
