import os
import sys
import numpy as np
from chainer import cuda
from chainer import serializers
from ResNet50 import ResNet
sys.path.append('../..')
from img_proc import Img_proc
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', '-id', type=str, default="../../../data/images/original/train2014/", 
        help="Directory images are saved")
parser.add_argument('--out_dir', '-od', type=str, default="../../../data/images/features/train2014/",
        help="Directory image features are saved")
parser.add_argument('--model', '-m', type=str, default="../../../data/models/cnn/ResNet50.model",
        help="model path to ResNet")
parser.add_argument('--gpu', '-g', type=int, default=-1, 
        help="GPU ID(if you don't use GPU set as -1)")
args = parser.parse_args()

#prepare image processer
img_proc = Img_proc(mean_type="imagenet")

model = ResNet()
serializers.load_hdf5(args.model, model)
model.train = False

#prepare GPU
if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
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
    features = model(img, "feature").data
    if args.gpu >= 0:
        features = cuda.to_cpu(features)
    np.savez('{0}{1}'.format(args.out_dir, name), features.reshape(2048))

#The dimention of it is 2048