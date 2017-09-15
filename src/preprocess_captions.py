import os
import json
import argparse
import numpy as np
from numpy import random


class Tokenizer(object):
    def __init__(self, args):lang
        self.args = args
        self.lang = self.args.lang

        if self.lang == 'jp':
            from janome.tokenizer import Tokenizer
            t = Tokenizer()
            self.segmenter= lambda sentence: list(token.surface for token in self.t.tokenize(sentence))

        elif self.lang == 'cn':
            import jieba
            self.segmenter= lambda sentence: list(jieba.cut(sentence))

        elif self.lang == 'en':
            import nltk
            self.nltk = nltk
            self.segmenter = lambda sentence: list(nltk.word_tokenize(sentence))
        
        elif self.lang == 'ch':
            self.sengment = lambda sentence: list(sentence)

    def pre_process(self, caption):
        if self.args.lower:
            caption = caption.strip().lower()
        if self.args.period and caption[-1] in ('.', '。'):
            caption = caption[0:-1]
        return self.segmenter(caption)


def words2ids(tokens, word_ids):
    return [ word_ids[token]['idx'] if token in word_ids else word_ids['<UNK>']['idx'] for token in tokens ]

def loadpickle(p_file):
    with open(p_file, 'rb') as f:
        data = pickle.load(f)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_train', '-it', type=str, default="../data/captions/converted/formatted_json_train_jp.pkl",
                        help="input formatted JSON train file"
    )
    parser.add_argument('--input_val', '-iv', type=str, default="../data/captions/converted/formatted_json_val_jp.pkl",
                        help="input formatted JSON val file"
    )
    parser.add_argument('--out_dir', '-od', type=str, default="../data/captions/processed",
                        help="output dir"
    )
    parser.add_argument('--out_file', '-of', type=str, default="dataset.pkl",
                        help="output file name"
    )
    parser.add_argument('--lang', '-l', type=str, choices=['jp', 'cn', 'en', 'ch'], default="jp",
                        help="dataset language you want to analyze"
    )
    parser.add_argument('--off', '-o', type=int, default=5,
                        help="min number of word frequency in words dict"
    )
    parser.add_argument('--lower', '-lw', type=bool, default=True,
                        help="lower all of the characters in captions"
    )
    parser.add_argument('--period', '-p', type=bool, default=True,
                        help="remove periods if captions has it"
    )
    parser.add_argument('--ratio', '-r', type=float, default=0,5,
                        help="The ratio of validation data")
    
    args = parser.parse_args()

    tokenizer = Tokenizer(args)

    formatted_train = loadpickle(args.input_train)
    formatted_val = loadpickle(args.input_val)

    train_data = []
    val_data = []
    test_data = []

    val_img_num = int(len(formatted_val) * args.ratio)

    #validation data and test data
    for i, img in enumerate(tqdm(formatted_val)):
        if 'tokenized_captions' in img:
            for j, caption in enumerate(img['tokenized_captions']):
                img[i]['tokenized_captions'][j] = caption.split()
        else:
            img[i]['tokenized_captions'] = []
            for j, caption in enumerate(img['captions']):
                img[i]['tokenized_captions'].append(tokenizer.pre_process(caption))

    random.seed(0)
    random.shuffle(formatted_val)
    val_data.append(formatted_val[:val_img_num])
    test_data.append(formatted_val[val_img_num:])
    train_data.append(formatted_train)

    img_idx = 0
    caption_idx = 0
    captions = []
    images = []

    word_counter = {}
    word_ids = {}

    for img in tqdm(formatted_train):
        if 'tokenized_captions' in img:
            for caption in enumerate(img['tokenized_captions']):
                caption_tokens = ['<S>']
                caption_tokens += caption.split()
                caption_tokens.append('</S>')
                captions.append({'img_idx': img_idx, 'caption': caption_tokens, 'caption_idx': caption_idx})
                caption_idx += 1
            #del img['captions']
            #del img['tokenized_captions']
            images.append({'file_path': img['file_path'], 'img_idx': img_idx})
            img_idx += 1

        else:
            for caption in enumerate(img['captions']):
                caption_tokens = ['<S>']
                caption_tokens += tokenizer.pre_process(caption)
                caption_tokens.append('</S>')
                captions.append({'img_idx': img_idx, 'caption': caption_tokens, 'caption_idx': caption_idx})
                caption_idx += 1
            #del img['captions']
            #del img['tokenized_captions']
            images.append({'file_path': img['file_path'], 'img_idx': img_idx})
            img_idx += 1
    
    #create vocabrary dictonary
    for caption in tqdm(captions):
        tokens = caption['caption']
        
        for token in tokens:
            if token in word_counter:
                word_counter[token] += 1
            else:
                word_counter[token] = 1

    print('total distinct words:' len(word_counter))
    print('top 30 frequent words:')
    sorted_word = sorted(word_counter.items(), key=lambda x: x[1], reverse=True)
    for word, freq in sorted_word[:20]:
        print('{0} - {1}'.format(word, freq))

    unknown = 0
    for word, num in tqdm(word_counter.items()):
        if num > args.off:
            word_ids[word] = {'freq':num, 'idx':len(word_ids)}
        else:
            unknown += 1
    word_ids['<UNK>'] = {'freq': unknown, 'idx': len(word_ids)}

    print('total distinct words more than {0} : {1}', args.off, len(word_ids))

    #encoding 
    for caption in tqdm(captions):
        caption['caption'] = words2ids(caption['caption'], word_ids)
        #caption['encoded_tokenized_caption'] = words2ids(caption['caption'], word_ids)

    output_dataset = {}
    output_dataset['train'] = {'images': images, 'captions': captions, 'word_ids': word_ids}
    output_dataset['val'] = val_data
    output_dataset['test'] = test_data
