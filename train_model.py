import sys
import os
import argparse
import numpy as np
import pickle
import chainer
import chainer.function as F
from chainer import cuda
from chainer.cuda import cupy as cp
from chainer import optimizers, serializers

sys.path.append('./src')
from Image2CaptionDecoder import Image2CaptionDecoder
from DataLoader import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', type=int, default=0,
                    help="set GPU ID (negative value means using CPU)")
parser.add_argument('--dataset', '-d', type=str, default="./data/captions/processed/dataset_STAIR_jp.pkl",
                    help="Path to preprocessed caption pkl file")
parser.add_argument('--img_feature_root', '-f', type=str, default="./data/images/features/ResNet50/")
parser.add_argument('--img_root', '-i', type=str, default="./data/images/original/",
                    help="Path to image files root")
parser.add_argument('--output_dir', '-od', type=str, default="./data/train_data/STAIR",
                    help="The directory to save model and log")
parser.add_argument('--preload', '-p', type=bool, default=False,
                    help="preload all image features onto RAM before trainig")
parser.add_argument('--epoch', type=int, default=10, 
                    help="The number of epoch")
parser.add_argument('--batch', type=int, default=256,
                    help="Mini batch size")
parser.add_argument('--hidden_dim', '-hd', type=int, default=512,
                    help="The number of hiden dim size in LSTM")
parser.add_argument('--img_feature_dim', '-fd', type=int, default=2048,
                    help="The number of image feature dim as input to LSTM")
parser.add_argument('--optimizer', '-opt', type=str, default="Adam", choices=['AdaDelta', 'AdaGrad', 'Adam', 'MomentumSGD', 'SGD', 'RMSprop'],
                    help="T")
parser.add_argument('--dropout_ratio', '-do', type=float, default=0.5,
                    help="Dropout ratio")
args = parser.parse_args()

#create save directories
if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)
    os.makedir(os.path.join(args.output_dir, 'models'))
    os.makedir(os.path.join(args.output_dir, 'optimizers'))
    os.makedir(os.path.join(args.output_dir, 'logs'))
    print('making some directories to ', args.output_dir)


#data preparation
print('loading preprocessed data...')

with open(args.dataset, 'rb') as f:
    data = pickle.load(f)

train_data = data['train']
val_data = data['val']
test_data = data['test']

#word dictionary
token2index = train_data['word_ids']

dataset = DataLoader(train_data, img_feature_root=args.img_feature_root, preload_features=args.preload, img_root=args.img_root)

#model preparation
model = Image2CaptionDecoder(vocab_size=len(token2index), hidden_dim=args.hidden_dim, img_feature_dim=args.img_feature_dim, dropout_ratio=args.dropout_ratio)

#cupy settings
if args.gpu:
    xp = cp
    cuda.get_device_from_id(args.gpu).use()
    model.to_gpu()
else:
    xp = np

opt = args.optimizer
'''
if opt == 'SGD':
    optimizer = optimizers.SGD()
elif opt == 'AdaDelta':
    optimizer = optimizers.AdaDelta()
elif opt == 'Adam':
    optimizer = optimizers.Adam()
'''
#optimizers
optimizer = optimizers.Adam()
optimizer.setup(model)

# configuration about training
total_epoch = args.epoch
batch_size = args.batch
data_size = dataset.get_caption_size()

# before training
print('-----configurations-----')
print('Total images: ', dataset.get_image_size())
print('Total captions:', dataset.get_caption_size())
print('epoch: ', total_epoch)
print('batch_size:', batch_size)
print('optimizer:', opt)
#print('Lerning rate: ', lerning_rate)


#start training

while dataset.epoch <= total_epoch:
    img_batch, cap_batch = dataset.get_batch(batch_size)
    
    if args.gpu:
        img_batch = cuda.to_gpu(img_batch, device=args.gpu)
        cap_batch = [ cuda.to_gpu(x, device=args.gpu) for x in cap_batch]

    #lstml inputs
    hx = xp.zeros((model, n_layers, len(x_batch), model.hidden_dim), dtype=xp.float32)
    cx = xp.zeros((model, n_layers, len(x_batch), model.hidden_dim), dtype=xp.float32)

    loss = model(hx, cx, cap_batch)

    model.cleargrads()
    model.backward()
    
    #update parameters
    optimizer.update()
