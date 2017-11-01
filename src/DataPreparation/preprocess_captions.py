import os
import argparse
import pickle
from numpy import random
from tqdm import tqdm

class Tokenizer(object):
    def __init__(self, args):
        self.args = args
        self.lang = self.args.lang

        if self.lang == 'jp':
            from janome.tokenizer import Tokenizer
            self.t = Tokenizer()
            self.segmenter= lambda sentence: list(token.surface for token in self.t.tokenize(sentence))

        elif self.lang == 'ch':
            import jieba
            self.segmenter= lambda sentence: list(jieba.cut(sentence))

        elif self.lang == 'en':
            import nltk
            self.nltk = nltk
            self.segmenter = lambda sentence: list(nltk.word_tokenize(sentence))
        
    def pre_process(self, caption):
        if self.args.lower:
            caption = caption.strip().lower()
        if self.args.period and caption[-1] in ('.', '。'):
            caption = caption[0:-1]
        return self.segmenter(caption)


def token2index(tokens, word_index):
    return [ word_index[token] if token in word_index else word_index['<UNK>'] for token in tokens ]

def load_pickle(p_file):
    with open(p_file, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pickle(out_data, p_file):
    with open(p_file, 'wb') as f:
        pickle.dump(out_data, f, pickle.HIGHEST_PROTOCOL)

def create_captions(formatted_json, tokenizer):
    
    img_idx = 0
    caption_idx = 0
    captions = []
    images = []
   
    for img in tqdm(formatted_json):
        if 'tokenized_captions' in img:
            for caption in img['tokenized_captions']:
                caption_tokens = ['<S>']
                caption_tokens += caption.split()
                caption_tokens.append('</S>')
                captions.append({'img_idx': img_idx, 'caption': caption_tokens, 'caption_idx': caption_idx})
                caption_idx += 1
            images.append({'file_path': img['file_path'], 'img_idx': img_idx})
            img_idx += 1

        else:
            for caption in img['captions']:
                caption_tokens = ['<S>']
                caption_tokens += tokenizer.pre_process(caption)
                caption_tokens.append('</S>')
                captions.append({'img_idx': img_idx, 'caption': caption_tokens, 'caption_idx': caption_idx})
                caption_idx += 1
            images.append({'file_path': img['file_path'], 'img_idx': img_idx})
            img_idx += 1

    return captions, images

def create_dict(captions, off):
    word_counter = {}
    word_index = { '<S>': 0, '</S>': 1, '<UNK>': 2, }
    
    #create vocabrary dictonary
    for caption in tqdm(captions):
        tokens = caption['caption']
        
        for token in tokens:
            if token in word_counter:
                word_counter[token] += 1
            else:
                word_counter[token] = 1
    
    print('total distinct words:', len(word_counter))
    print('top 30 frequent words:')
    sorted_word = sorted(word_counter.items(), key=lambda x: x[1], reverse=True)
    for word, freq in sorted_word[:30]:
        print('{0} - {1}'.format(word, freq))

    #create dict by cuttinf off some words
    for word, num in tqdm(word_counter.items()):
        if num > off:
            if word not in word_index:
                word_index[word] = len(word_index)
    
    print('total distinct words more than {0} : {1}'.format(args.off, len(word_index)))

    return word_index

def encode_captions(captions, word_index):
    for caption in tqdm(captions):
        caption['caption'] = token2index(caption['caption'], word_index)
    
    return captions

def make_dataset_bleu(formatted_data, tokenizer):
    for img in tqdm(formatted_data):
        if 'tokenized_aptions' in img:
            for i, caption in enumerate(img['tokenized_captions']):
                img['tokenized_captions'][i] = caption.split()
        else:
            img['tokenized_captions'] = []
            for caption in img['captions']:
                img['tokenized_captions'].append(tokenizer.pre_process(caption))
    
    return formatted_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_train', '-it', type=str, default=os.path.join('..', '..', 'data', 'captions', 'converted', 'formatted_json_train_jp.pkl'),
                        help="input formatted JSON train file"
    )
    parser.add_argument('--exist_val', '-ev', action='store_true',
                        help="exist validation file or not"
    )
    parser.add_argument('--input_val', '-iv', type=str, default=os.path.join('..', '..', 'data', 'captions', 'converted', 'formatted_json_val_jp.pkl'),
                        help="input formatted JSON val file"
    )
    parser.add_argument('--output_dataset_path', '-odap', type=str, default=os.path.join('..', '..', 'data', 'captions', 'processed', 'dataset_STAIR_jp.pkl'),
                        help="output file name"
    )
    parser.add_argument('--output_dataset_bleu_path', '-odb', type=str, default=os.path.join('..', '..', 'data', 'captions', 'blue', 'dataset_STAIR_jp_bleu.pkl'),
                        help="output dataset path of validation and test data for calculating bleu score")
    parser.add_argument('--output_dict_path', '-odip', type=str, default=os.path.join('..', '..', 'data', 'vocab_dict', 'dcit_STAIR_jp_train.pkl'),
                        help="output file name"
    )
    parser.add_argument('--lang', '-l', type=str, choices=['jp', 'cn', 'en', 'ch'], default="jp",
                        help="dataset language you want to analyze"
    )
    parser.add_argument('--off', '-o', type=int, default=5,
                        help="min number of word frequency in words dict"
    )
    parser.add_argument('--lower', '-lw', action='store_true',
                        help="lower all of the characters in captions"
    )
    parser.add_argument('--period', '-p', action='store_true',
                        help="remove periods if captions has it"
    )
    parser.add_argument('--ratio', '-r', type=float, default=0.5,
                        help="The ratio of validation data"
    )
    args = parser.parse_args()

    output_dataset = {}
    tokenizer = Tokenizer(args)
    
    formatted_train = load_pickle(args.input_train)
    train_captions, train_images = create_captions(formatted_train, tokenizer)
    word_index = create_dict(train_captions, args.off)
    train_captions = encode_captions(train_captions, word_index)
    output_dataset['train'] = {'images': train_images, 'captions': train_captions}
    
    if args.exist_val:
        formatted_val = load_pickle(args.input_val)
        
        random.seed(0)
        random.shuffle(formatted_val)
        val_img_num = int(len(formatted_val) * args.ratio)
        val_data = formatted_val[:val_img_num]
        test_data = formatted_val[val_img_num:]
        
        val_captions, val_images = create_captions(val_data, tokenizer) 
        val_captions = encode_captions(val_captions, word_index)
        
        test_captions, test_images = create_captions(test_data, tokenizer) 
        test_captions = encode_captions(test_captions, word_index)

        output_dataset['val'] = { 'images': val_images, 'captions': val_captions }
        output_dataset['test'] = { 'images': test_images, 'captions': test_captions }

        #bleu
        dataset_bleu = {}
        formatted_val = make_dataset_bleu(val_data, tokenizer) 
        formatted_test = make_dataset_bleu(test_data, tokenizer)
        dataset_bleu['val'] = formatted_val
        dataset_bleu['test'] = formatted_test

    output_dataset['word_index'] = word_index
    output_dict = word_index

    save_pickle(output_dataset, args.output_dataset_path)
    save_pickle(output_dict, args.output_dict_path)
    save_pickle(dataset_bleu, args.output_dataset_bleu_path)
