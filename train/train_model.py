import sys
import os
import math
import argparse
import numpy as np
import pickle
import chainer
from chainer import cuda
from chainer.cuda import cupy as cp
from chainer import optimizers, serializers

sys.path.append('../src')
from Image2CaptionDecoder import Image2CaptionDecoder
from DataLoader import DataLoader

from slack_notification import post_slack
import ENV

#separate dataset and dictionary file to use less memory
#prepare validation and test data 

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', type=int, default=0,
                    help="set GPU ID (negative value means using CPU)")
parser.add_argument('--dataset', '-d', type=str, default=os.path.join('..', 'data', 'captions', 'processed', 'dataset_STAIR_jp.pkl'),
                    help="Path to preprocessed caption pkl file")
parser.add_argument('--img_feature_root', '-f', type=str, default=os.path.join('..', 'data', 'images', 'features', 'ResNet50'),
                    help="Path to image feature root")
parser.add_argument('--img_root', '-i', type=str, default=os.path.join('.. ', 'data', 'images', 'original'),
                    help="Path to image files root")
parser.add_argument('--output_dir', '-od', type=str, default=os.path.join('..', 'data', 'train_data'),
                    help="The directory to save model and log")
parser.add_argument('--preload', '-p', action='store_true',
                    help="preload all image features onto RAM before trainig")
parser.add_argument('--epoch', type=int, default=100, 
                    help="The number of epoch")
parser.add_argument('--batch_size', type=int, default=256,
                    help="Mini batch size")
parser.add_argument('--hidden_dim', '-hd', type=int, default=512,
                    help="The number of hiden dim size in LSTM")
parser.add_argument('--img_feature_dim', '-fd', type=int, default=2048,
                    help="The number of image feature dim as input to LSTM")
parser.add_argument('--optimizer', '-opt', type=str, default="Adam", choices=['AdaDelta', 'AdaGrad', 'Adam', 'MomentumSGD', 'NesterovAG', 'RMSprop', 'RMSpropGraves', 'SGD', 'SMORMS3'],
                    help="Type of iptimizers")
parser.add_argument('--dropout_ratio', '-do', type=float, default=0.5,
                    help="Dropout ratio")
parser.add_argument('--n_layers', '-nl', type=int, default=1,
                    help="The number of layers")
parser.add_argument('--L2norm', '-l2', type=float, default=1.0,
                    help="L2 norm send to gradientclip")
parser.add_argument('--load_model', '-lm', type=int, default=0,
                    help="At which epoch you want to restart training(0 means training from zero)")
parser.add_argument('--slack', '-sl', action='store_true',
                    help="Notification to slack")
parser.add_argument('--validation', '-val', action='store_true',
                    help="exist validation file and run validation test")
args = parser.parse_args()

#create save directories
if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)
    os.mkdir(os.path.join(args.output_dir, 'models'))
    os.mkdir(os.path.join(args.output_dir, 'optimizers'))
    os.mkdir(os.path.join(args.output_dir, 'logs'))
    os.mkdir(os.path.join(args.output_dir, 'plots'))
    print('making some directories to ', args.output_dir)

#data preparation
print('loading preprocessed data...')

with open(args.dataset, 'rb') as f:
    data = pickle.load(f)

#word dictionary

dataset = DataLoader(data, img_feature_root=args.img_feature_root, preload_features=args.preload, img_root=args.img_root, exist_test = args.validation)

#model preparation
model = Image2CaptionDecoder(vocab_size=dataset.dict_size, hidden_dim=args.hidden_dim, img_feature_dim=args.img_feature_dim, dropout_ratio=args.dropout_ratio, n_layers=args.n_layers)

#cupy settings
if args.gpu >= 0:
    xp = cp
    cuda.get_device_from_id(args.gpu).use()
    model.to_gpu()
else:
    xp = np

#optimizers
opt = args.optimizer
grad_clip = args.L2norm

if opt == 'SGD':
    optimizer = optimizers.SGD()
elif opt == 'AdaDelta':
    optimizer = optimizers.AdaDelta()
elif opt == 'Adam':
    optimizer = optimizers.Adam()
elif opt == 'AdaGrad':
    optimizer = optimizers.AdaGrad()
elif opt == 'MomentumSGD':
    optimizer = optimizers.MomentumSGD()
elif opt == 'NesterovAG':
    optimizer = optimizers.NesterovAG()
elif opt == 'RMSprop':
    optimizer = optimizers.RMSprop()
elif opt == 'RMSpropGraves':
    optimizer = optimizers.RMSpropGraves()
elif opt == 'SMORMS3':
    optimizer = optimizers.SMORMS3()

optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.GradientClipping(grad_clip))

# configuration about training
total_epoch = args.epoch
batch_size = args.batch_size
train_caption_size = dataset.num_train_captions
val_caption_size = dataset.num_val_captions
total_iteration = math.ceil(train_caption_size / batch_size)
train_img_size = dataset.num_train_images
val_img_size = dataset.num_val_images
hidden_dim = args.hidden_dim
num_layers = args.n_layers
train_loss = 0
train_acc = 0
val_loss = 0
val_acc = 0
iteration = 0
l2norm = grad_clip
output_dir = os.path.join(args.output_dir, os.path.splitext(os.path.basename(args.dataset))[0] + '_' + str(batch_size) + '_' + str(opt))

sen_title = '-----configurations-----'
sen_gpu = 'GPU ID: ' + str(args.gpu)
sen_train_img = 'Total train images: ' + str(train_img_size)
sen_train_cap = 'Total train captions: ' + str(train_caption_size)
sen_val_img = 'Total val images: ' + str(val_img_size)
sen_val_cap = 'Total val captions: ' + str(val_caption_size)
sen_epoch = 'Total epoch: ' + str(total_epoch)
sen_batch = 'Batch size: ' + str(batch_size)
sen_hidden = 'The number of hidden dim: ' + str(hidden_dim)
sen_LSTM = 'The number of LSTM layers: ' + str(num_layers)
sen_optimizer = 'Optimizer: ' + str(opt)
sen_learnning = 'Learning rate: '
sen_l2norm = 'L2norm: ' + str(l2norm)

sen_conf = sen_title + '\n' + sen_gpu + '\n' + sen_train_img + '\n' + sen_train_cap + '\n' + sen_val_img + '\n' + sen_val_cap + '\n' + sen_epoch + '\n' + sen_batch + '\n' + sen_hidden + '\n' + sen_LSTM + '\n' + sen_optimizer + '\n' + sen_l2norm + '\n'

#create save directories
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
    os.mkdir(os.path.join(output_dir, 'models'))
    os.mkdir(os.path.join(output_dir, 'optimizers'))
    os.mkdir(os.path.join(output_dir, 'logs'))
    print('making some directories to ', output_dir)
    with open(os.path.join(output_dir, 'logs', 'logs.txt'), 'w') as f:
        sen_log_title = 'epoch,train/loss,train/acc,val/loss,val/acc'
        f.write(sen_log_title)

# before training
print(sen_conf)

with open(os.path.join(output_dir, 'logs', 'configurations.txt'), 'w') as f:
    f.write(sen_conf)

# load model
if args.load_model:
    cap_model_path = os.path.join(output_dir, 'models', 'caption_model' + str(args.load_model) + '.model')
    opt_model_path = os.path.join(output_dir, 'optimizers', 'optimizer' + str(args.load_model) + '.model')
    serializers.load_hdf5(cap_model_path, model)
    serializers.load_hdf5(opt_model_path, optimizer)
    
    dataset.epoch = args.load_model + 1

#start training

print('\nepoch 1')

while dataset.now_epoch <= total_epoch:
    
    model.cleargrads()
    
    now_epoch = dataset.now_epoch
    img_batch, cap_batch = dataset.get_batch_train(batch_size)
    
    if args.gpu >= 0:
        img_batch = cuda.to_gpu(img_batch, device=args.gpu)
        cap_batch = [ cuda.to_gpu(x, device=args.gpu) for x in cap_batch]

    #lstml inputs
    hx = xp.zeros((num_layers, batch_size, model.hidden_dim), dtype=xp.float32)
    cx = xp.zeros((num_layers, batch_size, model.hidden_dim), dtype=xp.float32)
    hx, cx = model.input_cnn_feature(hx, cx, img_batch)
    loss, acc = model(hx, cx, cap_batch)

    loss.backward()
    #loss.unchain_backward()
    
    #update parameters
    optimizer.update()

    train_loss += loss.data * batch_size
    train_acc += acc.data * batch_size
    iteration += 1

    print('epoch: {0} iteration: {1}, loss: {2}, acc: {3}'.format(now_epoch, str(iteration) + '/' + str(total_iteration), round(float(loss.data), 10), round(float(acc.data), 10)))
    
    if now_epoch is not dataset.now_epoch:
        mean_train_loss = train_loss / train_caption_size
        mean_train_acc = train_acc / train_caption_size
        
        #validation
        dataset.continue_val = True
        while dataset.continue_val:
            img_batch_val, cap_batch_val = dataset.get_batch_val(batch_size)
            
            if args.gpu >= 0:
                img_batch_val = cuda.to_gpu(img_batch_val, device=args.gpu)
                cap_batch_val = [ cuda.to_gpu(x, device=args.gpu) for x in cap_batch_val]
        
            hx = xp.zeros((num_layers, batch_size, model.hidden_dim), dtype=xp.float32)
            cx = xp.zeros((num_layers, batch_size, model.hidden_dim), dtype=xp.float32)
            hx, cx = model.input_cnn_feature(hx, cx, img_batch_val)
            loss, acc = model(hx, cx, cap_batch_val)

            val_loss += loss.data * batch_size
            val_acc += acc.data * batch_size
        
        mean_val_loss = val_loss / val_caption_size
        mean_val_acc = val_acc / val_caption_size

        print('\nepoch {0} result'.format(now_epoch))
        print('train_loss: {0} train_acc: {1} val_loss: {2} val_acc: {3}'.format(round(float(mean_train_loss), 10), round(float(mean_train_acc), 10), round(float(mean_val_loss), 10), round(float(mean_val_acc), 10)))
        print('\nepoch ', now_epoch)

        serializers.save_hdf5(os.path.join(output_dir, 'models', 'caption_model' + str(now_epoch) + '.model'), model)
        serializers.save_hdf5(os.path.join(output_dir, 'optimizers', 'optimizer' + str(now_epoch) + '.model'), optimizer)
        
        with open(os.path.join(output_dir, 'logs', 'logs.txt'), 'a') as f:
            log_sen = str(now_epoch) + ',' + str(mean_train_loss) + ',' + str(mean_train_acc) + ',' + str(mean_val_loss) + ',' + str(mean_val_acc) + '\n'
            f.write(log_sen)

        if args.slack:
            name = output_dir
            if name[-1] == '/':
                name =name[:-1]
            name = os.path.basename(name)
            text = 'epoch: ' + str(now_epoch) + '\ntrain/loss: ' + str(mean_train_loss) + '\natrain/acc:' + str(mean_train_acc) + '\nval/loss: ' + str(mean_val_loss) + '\nval/acc: ' + str(mean_val_acc)
            #ENV.POST_URL is set at ENV.py
            post_slack(ENV.POST_URL, name, text)
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0
        iteration = 0
