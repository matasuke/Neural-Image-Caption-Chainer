import sys
import numpy as np
import pickle
import argparse
from copy import deepcopy
import chainer
from chainer import cuda
import chainer.functions as F
from chainer import cuda
from chainer import serializers

from img_proc import Img_proc
from Image2CaptionDecoder import Image2CaptionDecoder

import heapq

class CaptionGenerator(object):
    def __init__(self, rnn_model_path, cnn_model_path, dict_path, cnn_model_type="ResNet", beamsize=3, depth_limit=50, gpu_id=-1, first_word="<S>", hidden_dim=512, mean="imagenet"):
        self.gpu_id = gpu_id
        self.beamsize = beamsize
        self.depth_limit = depth_limit
        self.img_proc = Img_proc(mean_type=mean)
        self.index2token = self.parse_dic(dict_path)

        if cnn_model_type == 'ResNet':
            sys.path.append('CNN/resnet/')
            from ResNet50 import ResNet
            self.cnn_model = ResNet()
        elif cnn_model_type == 'VGG16':
            sys.path.append('CNN/VGG/')
            from VGG16 import VGG16
            self.cnn_model = VGG16()
        elif cnn_model_type == 'AlexNet':
            sys.path.append('CNN/AlexNet')
            from AlexNet import AlexNet
            self.cnn_model = AlexNet()

        serializers.load_hdf5(cnn_model_path, self.cnn_model)
        self.cnn_model.train = False

        self.rnn_model = Image2CaptionDecoder(len(self.token2index), hidden_dim=hidden_dim)
        if len(rnn_model_path) > 0:
            serializers.load_hdf5(rnn_model_path, self.rnn_model)
        self.rnn_model.train = False

        self.first_word = first_word
        
        #Gpu configuration
        #self.xp
        global xp
        if self.gpu_id >= 0:
            xp = cuda.cupy
            cuda.get_device(gpu_id).use()
            self.cnn_model.to_gpu()
            self.rnn_model.to_gpu()
        else:
            xp=np

    def parse_dic(self, dict_path):
        with open(dict_path, 'rb') as f:
            self.token2index = pickle.load(f)['train']['word_ids']

        return { v:k for k, v in self.token2index.items() }

    def successor(self, current_state):
        word=[xp.array([current_state["path"][-1]],dtype=xp.int32)]
        hx = current_state['hidden']
        cx = current_state['cell']
    
        #predict next word
        with chainer.using_config('train', False):
            hy, cy, next_words = self.rnn_model(hx, cx, word)
        
        word_dist = F.softmax(next_words[0]).data[0]
        k_best_next_sentences = []
        for i in range(self.beamsize):
            next_word_idx = int(xp.argmax(word_dist))
            k_best_next_sentences.append(
                    {
                        "hidden": hy,
                        "cell": cy,
                        "path": deepcopy(current_state['path']) + [next_word_idx],
                        "cost": current_state['cost'] - xp.log(word_dist[next_word_idx])
                    }
                )
            word_dist[next_word_idx] = 0

        return hy, cy, k_best_next_sentences

    def beam_search(self, initial_state):
        
        found_paths = []
        top_k_states = [initial_state]
        while (len(found_paths) < self.beamsize):
            new_top_k_states = []
            for state in top_k_states:
                hy, cy, k_best_next_states = self.successor(state)
                for next_state in k_best_next_states:
                    new_top_k_states.append(next_state)
            selected_top_k_states=heapq.nsmallest(self.beamsize, new_top_k_states, key=lambda x: x['cost'])

            top_k_states=[]
            for state in selected_top_k_states:
                if state['path'][-1] == self.token2index['</S>'] or len(state['path']) == self.depth_limit:
                    found_paths.append(state)
                else:
                    top_k_states.append(state)

        return sorted(found_paths, key=lambda x: x['cost'])

    def generate(self, img_file_path):
        img = self.img_proc.load_img(img_file_path)
        return self.generate_from_img(img)

    def generate_from_img_feature(self, img_feature):
        if self.gpu_id >= 0:
            img_feature = cuda.to_gpu(img_feature)

        batch_size = 1
        hx = xp.zeros((self.rnn_model.n_layers, batch_size, self.rnn_model.hidden_dim), dtype=xp.float32)
        cx = xp.zeros((self.rnn_model.n_layers, batch_size, self.rnn_model.hidden_dim), dtype=xp.float32)
        
        with chainer.using_config('train', False):
            hy, cy = self.rnn_model.input_cnn_feature(hx, cx, img_feature)

        initial_state = {
                "hidden": hy,
                "cell": cy,
                "path": [self.token2index[self.first_word]],
                "cost": 0,
                }

        captions = self.beam_search(initial_state)

        caption_candidates = []
        for caption in captions:
            sentence = [self.index2token[word_idx] for word_idx in caption['path']]
            log_likelihood = -float(caption['cost']) #negative log likelihood
            caption_candidates.append({'sentence': sentence, 'log_likelihood': log_likelihood})

        return caption_candidates

    def generate_from_img(self, img_array):
        if self.gpu_id >= 0:
            img_array = cuda.to_gpu(img_array)
        with chainer.using_config('train', False):
            img_feature = self.cnn_model(img_array, 'feature').data.reshape(1, 1, 2048)

        return self.generate_from_img_feature(img_feature)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--rnn_model_path', '-rm', type=str, default="../data/models/rnn/caption_model_STAIR.model",
                        help="RNN model path")
    parser.add_argument('--cnn_model_path', '-cm', type=str, default="../data/models/cnn/ResNet50.model",
                        help="CNN model path")
    parser.add_argument('--dict_path', '-d', type=str, default="../data/captions/processed/dataset_STAIR_jp.pkl",
                        help="Dictionary path")
    parser.add_argument('--cnn_model_type', '-ct', type=str, choices=['AdaDelta', 'AdaGrad', 'Adam', 'MomentumSGD', 'NesterovAG', 'RMSprop', 'RMSpropGraves', 'SGD', 'SMORMS3'], default="ResNet",
                        help="CNN model type")
    parser.add_argument('--beamsize', '-b', type=int, default=3,
                        help="beamsize")
    parser.add_argument('--depth_limit', '-dl', type=int, default=50,
                        help="max limit of generating tokens when constructing captions")
    parser.add_argument('--gpu', '-g', type=int, default=0, 
                        help="set GPU ID(negative value means using CPU)")
    parser.add_argument('--first_word', '-fw', type=str, default='<S>',
                        help="set first word")
    parser.add_argument('--hidden_dim', '-hd', type=int, default=512,
                        help="dimension of hidden layers")
    parser.add_argument('--mean', '-m', type=str, choices=['imagenet'], default='imagenet',
                        help="method to preproces images")
    parser.add_argument('--img', '-i', type=str, default='../sample_imgs/sample_img2.jpg',
                        help="path to test image (default is set as sample_img1.jpg)")
    args = parser.parse_args()
    
    
    '''
    have to fixt dict_path for let it use vocaburary table
    now it can't be used. you can just use dataset.pkl file, which contains word2ids.
    '''

    xp = np
    caption_generator = CaptionGenerator(
            rnn_model_path = args.rnn_model_path,
            cnn_model_path = args.cnn_model_path,
            dict_path = args.dict_path,
            cnn_model_type=args.cnn_model_type,
            beamsize = args.beamsize,
            depth_limit = args.depth_limit,
            gpu_id = args.gpu,
            first_word=args.first_word,
            hidden_dim=args.hidden_dim,
            mean=args.mean
        )

    '''
    batch size is set as 1
    I'll fix it later
    '''

    '''
    batch_size = 1
    hx=xp.zeros((caption_generator.rnn_model.n_layers, batch_size, caption_generator.rnn_model.hidden_dim), dtype=xp.float32)
    cx=xp.zeros((caption_generator.rnn_model.n_layers, batch_size, caption_generator.rnn_model.hidden_dim), dtype=xp.float32)
    img=caption_generator.image_loader.load("../sample_imgs/COCO_val2014_000000185546.jpg")
    image_feature=caption_generator.cnn_model(img, "feature").data.reshape(1,1,2048))

    hy,cy = caption_generator.rnn_model.input_cnn_feature(hx, cx, img_feature)
    initial_state = {
            "hidden": hy,
            "cell": cy,
            "path": [caption_generator.token2index['<S>']]
            "cost": 0
        }

    '''
    captions = caption_generator.generate(args.img)
    for i, caption in enumerate(captions):
        print('caption{0}: {1}'.format(i, caption['sentence']))
        print('log: ', caption['log_likelihood'])
