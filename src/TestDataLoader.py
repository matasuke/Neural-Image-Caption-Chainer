from DataLoader import DataLoader
import argparse
import pickle
import numpy as np

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

for i in range(0, args.epoch):
    img_batch, cap_batch = dataset.get_batch(batch_size, raw_img=False)

    for img, words in zip(img_batch, cap_batch):
        print('img', img)
        print('words', words)

'''
    print(np.shape(img_batch))
    print(np.shape(cap_batch))
    
    if i <= 1:
        for img, words in zip(img_batch, cap_batch):
            print(np.shape(img)) 
            print(np.shape(words)) 
'''
