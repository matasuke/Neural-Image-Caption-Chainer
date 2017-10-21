import sys
import os
import math
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
parser.add_argument('--n_layers', '-nl', type=int, default=1,
                    help="The number of layers")
args = parser.parse_args()

#create save directories
if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)
    os.mkdir(os.path.join(args.output_dir, 'models'))
    os.mkdir(os.path.join(args.output_dir, 'optimizers'))
    os.mkdir(os.path.join(args.output_dir, 'logs'))
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
model = Image2CaptionDecoder(vocab_size=len(token2index), hidden_dim=args.hidden_dim, img_feature_dim=args.img_feature_dim, dropout_ratio=args.dropout_ratio, n_layers=args.n_layers)

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
caption_size = dataset.caption_size
total_iteration = math.ceil(caption_size / batch_size)
img_size = dataset.img_size
num_layers = args.n_layers
sum_loss = 0
iteration = 0

# before training
print('-----configurations-----')
print('Total images: ', img_size)
print('Total captions:', caption_size)
print('total epoch: ', total_epoch)
print('batch_size:', batch_size)
print('The number of LSTM layers:', num_layers)
print('optimizer:', opt)
#print('Lerning rate: ', lerning_rate)


#start training

print('\nepoch 1')

while dataset.epoch <= total_epoch:
    
    model.cleargrads()
    
    now_epoch = dataset.now_epoch
    img_batch, cap_batch = dataset.get_batch(batch_size)
    
    if args.gpu:
        img_batch = cuda.to_gpu(img_batch, device=args.gpu)
        cap_batch = [ cuda.to_gpu(x, device=args.gpu) for x in cap_batch]

    #lstml inputs
    hx = xp.zeros((num_layers, batch_size, model.hidden_dim), dtype=xp.float32)
    cx = xp.zeros((num_layers, batch_size, model.hidden_dim), dtype=xp.float32)

    loss = model(hx, cx, cap_batch)

    loss.backward()
    
    #update parameters
    optimizer.update()

    sum_loss += loss.data * batch_size
    iteration += 1
    
    print('epoch: {0} iteration: {1}, loss: {2}'.format(now_epoch, str(iteration) + '/' + str(total_iteration), round(float(loss.data), 5)))
    if now_epoch is total_epoch:
        
        mean_loss = sum_loss / caption_size

        print('\nepoch {0} result/n', now_epoch-1)
        print('epoch: {0} iteration: {1}, loss: {2:.5f}'.format(now_epoch, str(iteration) + '/' + str(total_iteration), round(float(mean_loss), 5)))
        print('\nepoch ', now_epoch)

        serializers.save_hdf5(os.path.join(args.output_dir, 'models', 'caption_model' + str(now_epoch) + '.model'), model)
        serializers.save_hdf5(os.path.join(args.output_dir, 'optimizers', 'optimizer' + str(now_epoch) + '.model'), optimizer)
        
        with open(os.path.join(args.output_dir, 'logs', 'mean_loss.txt'), 'a') as f:
            f.write(str(mean_loss) + '\n')
        sum_loss = 0
        iteration = 0
