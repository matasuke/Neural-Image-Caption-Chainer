import sys
import os
import argparse
import numpy as np
import pickle
import chainer
import chainer.functions as F
from chainer import cuda
from chainer import Function, FunctionSet, Variable, optimizers, serializers

sys.path.append('./src')
from Image2CaptionDecoder import Image2CaptionDecoder
from DataLoader import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default=0, type=int, 
                    help="set GPU ID(if use CPU, set as -1)")
parser.add_argument('--savedir', default="./data/train_data/STAIR", type=str,
                    help="directory to save models and log")

'''
parser.add_argument('--vocab', '-v', default="./data/captions/processed/dataset_STAIR_jp.pkl", type=str,
                    help="path to the vocabrary path")
parser.add_argument('--captions', default="./data/captions/processed/dataset_STAIR_jp.pkl", type=str,
                    help="path to preprocessed caption pkl file")
'''
parser.add_argument('--dataset', '-d', default="./data/captions/processed/dataset.pkl", type=str,
                    help="path to dataset file")
parser.add_argument('--img_feature_root', '-f', default="./data/images/features/ResnNet50/", type=str,
                    help="path to CNN feature")
parser.add_argument('--filename_img_id', '-id', default='./data/images/original/', type=str,
                    help="image id is filename")
#parser.add_argument('--filename_img_id', '-id', default='./data/images/original/', type=str,
#                    help="image id is filename")
parser.add_argument('--preload', default=False, type=bool,
                    help="preload all image features onto RAM")
parser.add_argument('--epoch', default=10, type=int,
                    help="the number of epoch")
parser.add_argument('--batch', '-b', default=128, type=int,
                    help="mini batch size")
parser.add_argument('--hidden', '-hd', default=512, type=int, 
                    help="the number of hidden size in LSTM")
args = parser.parse_args()

#save dir
if not os.path.isdir(args.savedir):
    os.makedirs(args.savedir)
    os.makedi(os.path.join(args.savedir, 'models'))
    os.makedi(os.path.join(args.savedir, 'optimizers'))
    os.makedi(os.path.join(args.savedir, 'logs'))
    print("make the save directory", args.savedir)

#prepare GPU
if args.gpu >= 0:
    xp = cuda.cupy
    cuda.get_device(args.gpu).use()
else:
    xp=np

#prepare data
#these lines have to be fixed
print('loading preprocessed trainig data...')

with open(args.dataset, 'rb') as f:
    data = pickle.load(f)

train_data = data['train']

#prepare words ids
token2index = train_data['word_ids']

dataset = DataLoader(train_data, img_feature_root=args.img_feature_root, preload_features=args.preload, img_root=args.filename_img_id)

#model preparation
print('preparing caption generation models and trainig process')
model = Image2CaptionDecoder(vocab_size=len(token2index), hidden_dim=args.hidden)

#gpu processing
if args.gpu >= 0:
    model.to_gpu()

#optimizers
optimizer = optimizers.Adam()
optimizer.setup(model)

# setting for training
batch_size = args.batch
grad_clip = 1.0
num_train_data = len(train_data)

#start training
sum_loss = 0
print('Training start')
iteration = 1

while(dataset.epoch <= args.epoch):
    optimizer.zero_grads()
    #model.zerograds()
    current_epoch = dataset.epoch
    img_feature, x_batch = dataset.get_batch(batch_size)
    
    if args.gpu >= 0:
        img_feature = cuda.to_gpu(img_feature, device=args.gpu)
        x_batch = [ cuda.to_gpu(x, device=args.gpu) for x in x_batch ]
    
    hx = xp.zeros((model.n_layers, len(x_batch), model.hidden_dim), dtype=xp.float32)
    cx = xp.zeros((model.n_layers, len(x_batch), model.hidden_dim), dtype=xp.float32)
    hx, cx = model.input_cnn_feature(hx, cx, img_feature)
    loss = model(hx, cx, x_batch)

    print(loss.data)
    with open(os.path.join(args.savedir, 'logs', 'loss.txt'), "a") as f:
        f.write(str(loss.data) + '\n')

    loss.backward()
    loss.unchain_backward()
    optimizer.clip_grads(grad_clip)
    optimizer.update()

    sum_loss += loss.data * batch_size
    iteration += 1

    if dataset.epoch - current_epoch > 0 or iteration > 10000:
        print('epoch:', dataset.epoch)
        serializers.save_hdf5(os.path.join(args.savedir, 'models', '/caption_model' + current_epoch + '.model'), model)
        serializers.save_hdf5(os.path.join(args.savedir, 'optimizers', 'optimizer' + current_epoch + '.model'), optimizer)

        mean_loss = sum_loss / num_train_data
        with open(os.path.join(args.savedir, 'logs', 'mean_loss.txt'),'a') as f:
            f.write(str(mean_loss) + '\n')
        sum_loss= 0
        iteration = 0
