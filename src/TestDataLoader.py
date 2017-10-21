from DataLoader import DataLoader
import argparse
import pickle
import numpy as np
from chainer import cuda

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', '-d', type=str, default="../data/captions/processed/dataset_STAIR_jp.pkl",
        help="Path to preprocessed caption pkl file")
parser.add_argument('--img_feature_root', type=str, default="../data/images/features/ResNet50/",
        help="Path to image feature root")
parser.add_argument('--img_root', type=str, default="../data/images/original/",
        help="Path to image file root")
parser.add_argument('--preload', '-p', type=bool, default=True,
        help="preload all image features onto RAM before trainig")
parser.add_argument('--epoch', '-e', type=int, default=20,
        help="The number of epoch")
parser.add_argument('--batch_size', '-b', type=int, default=256,
        help="batch size")
args = parser.parse_args()

with open(args.dataset, 'rb') as f:
    data = pickle.load(f)

train_data = data['train']
val_data = data['val']
test_data = data['test']

token2index = train_data['word_ids']

dataset = DataLoader(train_data, img_feature_root=args.img_feature_root, preload_features = args.preload, img_root=args.img_root)

batch_size = args.batch_size

cuda.get_device_from_id(0).use()

iteration = 0

while dataset.now_epoch <= args.epoch:
    
    now_epoch = dataset.now_epoch
    img_batch, cap_batch = dataset.get_batch(batch_size)
    
    img_batch = cuda.to_gpu(img_batch, device=0)
    cap_batch = [ cuda.to_gpu(x, device=0) for x in cap_batch]
    
    if dataset.now_epoch < 3:
        print(np.shape(img_batch))
        print(np.shape(cap_batch))

    iteration += 1

    if now_epoch is not dataset.now_epoch:
        print('new epoch')
        iteration = 0

'''
    for img, words in zip(img_batch, cap_batch):
        print('img', img)
        print('words', words)
'''
'''
    print(np.shape(img_batch))
    print(np.shape(cap_batch))
    
    if i <= 1:
        for img, words in zip(img_batch, cap_batch):
            print(np.shape(img)) 
            print(np.shape(words)) 
'''
