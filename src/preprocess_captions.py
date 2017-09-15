import os
import json
import argparse
import numpy as np
from numpy.random import shuffle


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


def words2ids(tokens):
    return [ word_ids[token] if token in word_ids else unknown_word for token in tokens ]

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

    list_sentences = {
            k: {} for k in ['train', 'val', 'test']
            }

    list_img_ids 
