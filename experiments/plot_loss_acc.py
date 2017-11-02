import os
import matplotlib.pyplot as plt
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="plot loss and acc of train and validation result")
parser.add_argument('--input_jp_path', '-ij', type=str, default=os.path.join('..', 'data', 'train_data', 'dataset_STAIR_jp_256_Adam', 'logs', 'logs.txt'),
                    help="input japanese log file")
parser.add_argument('--input_en_path', '-ie', type=str, default=os.path.join('..', 'data', 'train_data', 'dataset_MSCOCO_en_256_Adam', 'logs', 'logs.txt'),
                    help="input english log file")
parser.add_argument('--input_ch_path', '-ic', type=str, default=os.path.join('..', 'data', 'train_data', 'dataset_MSCOCO_ch_mt_256_Adam', 'logs', 'logs.txt'),
                    help="input chinese log file")
parser.add_argument('--output_path', '-o', type=str, default=os.path.join('..', 'images'))
args = parser.parse_args()


jdf = pd.read_csv(args.input_jp_path, sep=',')
edf = pd.read_csv(args.input_en_path, sep=',')
cdf = pd.read_csv(args.input_ch_path, sep=',')

#epoch = list(range(0, len(df['epoch'])))
epoch = len(jdf['epoch'])
jp_train_loss = jdf['train/loss']
jp_train_acc = jdf['train/acc']

en_train_loss = edf['train/loss']
en_train_acc = edf['train/acc']

ch_train_loss = cdf['train/loss']
ch_train_acc = cdf['train/acc']

#plot accuracy
plt.plot(jp_train_acc, label="train acc jp")
plt.plot(en_train_acc, label="train acc en")
plt.plot(ch_train_acc, label="train acc ch")
plt.title('model accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.xlim(0, epoch)
plt.legend(loc='lower right')
plt.savefig(os.path.join(args.output_path, 'acc.png'))
plt.close()

#plot loss
plt.plot(jp_train_loss, label="train loss jp")
plt.plot(en_train_loss, label="train loss en")
plt.plot(ch_train_loss, label="train loss ch")
plt.title('model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.xlim(0, epoch)
plt.legend(loc='upper right')
plt.savefig(os.path.join(args.output_path, 'loss.png'))
plt.close()
