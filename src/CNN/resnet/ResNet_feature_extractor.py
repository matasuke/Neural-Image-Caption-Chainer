import os
from pathlib import Path
import numpy as np
import chainer
from chainer import cuda
from chainer import serializers
from .ResNet50 import ResNet
from src.img_proc import Img_proc
import argparse

DEFAULT_INPUT_DIR = Path('data/images/original/train2014')
DEFAULT_OUTPUT_DIR = Path('data/images/features/ResNet50/train2014')
DEFAULT_MODEL_DIR = Path('data/models/cnn/ResNet50.model')

parser = argparse.ArgumentParser()
parser.add_argument(
    '--input_dir', '-id', type=str,
    default=DEFAULT_INPUT_DIR.as_posix(),
    help="Directory images are saved"
)
parser.add_argument(
    '--output_dir', '-od', type=str,
    default=DEFAULT_OUTPUT_DIR.as_posix(),
    help="Directory image features are saved"
)
parser.add_argument(
    '--model', '-m', type=str,
    default=DEFAULT_MODEL_DIR.as_posix(),
    help="model path to ResNet"
)
parser.add_argument(
    '--gpu', '-g', type=int, default=-1,
    help="GPU ID(if you don't use GPU set as -1)"
)
args = parser.parse_args()

# make save dirs
if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)

# prepare image processer
img_proc = Img_proc(mean_type="imagenet")

model = ResNet()
serializers.load_hdf5(args.model, model)

# prepare GPU
if args.gpu >= 0:
    cuda.get_device_from_id(args.gpu).use()
    model.to_gpu()

# get the list of images
img_files = os.listdir(args.input_dir)

for i, path in enumerate(img_files):
    name, ext = os.path.splitext(path)
    print(i, path)
    img_path = os.path.join(args.input_dir, path)
    img = img_proc.load_img(img_path)

    if args.gpu >= 0:
        img = cuda.to_gpu(img, device=args.gpu)

    with chainer.using_config('train', False):
        features = model(img, "feature").data

    if args.gpu >= 0:
        features = cuda.to_cpu(features)

    np.savez('{0}'.format(os.path.join(args.output_dir, name)), features.reshape(2048))
