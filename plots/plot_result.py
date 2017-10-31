import os
import matplotlib.pyplot as plt
import pandas as pd
import argparse


parser = argparse.ArgumentParser(description="plot loss and acc of train and validation result")
parser.add_argument('--input_path', '-i', type=str, default=os.path.join('..', 'data', 'train_data', 'logs', 'logs.txt'),
                    help="input log file")
parser.add_argument('--output_path', '-o', type=str, default=os.path.join('..', 'data', 'train_data', 'plots'))
args = parser.parse_args()


df = pd.read_csv(args.input_path, sep=',')

#epoch = list(range(0, len(df['epoch'])))
epoch = len(df['epoch'])
train_loss = df['train/loss']
val_loss = df['val/loss']
train_acc = df['train/acc']
val_acc = df['val/acc']

#plot accuracy
plt.plot(train_acc, label="train acc")
plt.plot(val_acc, label="val acc")
plt.title('model accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.xlim(0, epoch)
plt.legend(loc='lower right')
plt.savefig(args.output_path + 'acc.png')
plt.close()

#plot loss
plt.plot(train_loss, label="train loss")
plt.plot(val_loss, label="val loss")
plt.title('model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.xlim(0, epoch)
plt.legend(loc='upper right')
plt.savefig(args.output_path + 'loss.png')
plt.close()
