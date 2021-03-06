import argparse
from pathlib import Path
import pickle
from numpy import random
from tqdm import tqdm
from typing import Dict

class Tokenizer(object):
    def __init__(self, args):
        self.args = args
        self.lang = self.args.lang

        if self.lang == 'jp':
            from janome.tokenizer import Tokenizer
            self.t = Tokenizer()
            self.segmenter = lambda sentence: list(token.surface for token in self.t.tokenize(sentence))

        elif self.lang == 'ch':
            import jieba
            self.segmenter = lambda sentence: list(jieba.cut(sentence))

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


def token2index(tokens, word_ids):
    return [word_ids[token] if token in word_ids else word_ids['<UNK>'] for token in tokens]

def load_pickle(p_file):
    with open(p_file, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pickle(out_data, p_file):
    with open(p_file, 'wb') as f:
        pickle.dump(out_data, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    DEFAULT_INPUT_TRAIN_PATH = Path('data/captions/converted/formatted_json_train_jp.pkl')
    DEFAULT_INPUT_VAL_PATH = Path('data/captions/converted/formatted_json_val_jp.pkl')
    DEFAULT_OUTPUT_PATH = Path('data/captions/processed/dataset_STAIR_jp.pkl')
    DEFAULT_VOCAB_OUTPUT_PATH = Path('data/vocab_dict/dict_STAIR_jp_train.pkl')

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_train', '-it', type=str,
        default=DEFAULT_INPUT_TRAIN_PATH.as_posix(),
        help="input formatted JSON train file"
    )
    parser.add_argument(
        '--exist_val', '-ev', action='store_true',
        help="exist validation file or not"
    )
    parser.add_argument(
        '--input_val', '-iv', type=str,
        default=DEFAULT_INPUT_VAL_PATH.as_posix(),
        help="input formatted JSON val file"
    )
    parser.add_argument(
        '--output_dataset_path', '-odap', type=str,
        default=DEFAULT_OUTPUT_PATH.as_posix(),
        help="output file name"
    )
    parser.add_argument(
        '--output_dict_path', '-odip', type=str,
        default=DEFAULT_VOCAB_OUTPUT_PATH.as_posix(),
        help="output file name"
    )
    parser.add_argument(
        '--lang', '-l', type=str, choices=['jp', 'cn', 'en', 'ch'], default="jp",
        help="dataset language you want to analyze"
    )
    parser.add_argument(
        '--off', '-o', type=int, default=5,
        help="min number of word frequency in words dict"
    )
    parser.add_argument(
        '--lower', '-lw', action='store_true',
        help="lower all of the characters in captions"
    )
    parser.add_argument(
        '--period', '-p', action='store_true',
        help="remove periods if captions has it"
    )
    parser.add_argument('--ratio', '-r', type=float, default=0.5,
                        help="The ratio of validation data")

    args = parser.parse_args()

    tokenizer = Tokenizer(args)

    formatted_train = load_pickle(args.input_train)

    if args.exist_val:
        formatted_val = load_pickle(args.input_val)

    img_idx = 0
    caption_idx = 0
    captions = []
    images = []

    word_counter: Dict[str, int] = {}
    word_ids = {
        '<S>': 0,
        '</S>': 1,
        '<UNK>': 2,
    }

    for img in tqdm(formatted_train):
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

    # create vocabrary dictonary
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

    for word, num in tqdm(word_counter.items()):
        if num > args.off:
            if word not in word_ids:
                word_ids[word] = len(word_ids)
            # word_ids[word] = {'freq':num, 'idx':len(word_ids)}

    print('total distinct words more than {0} : {1}'.format(args.off, len(word_ids)))

    for caption in tqdm(captions):
        caption['caption'] = token2index(caption['caption'], word_ids)

    # validation data and test data
    if args.exist_val:
        for img in tqdm(formatted_val):
            if 'tokenized_captions' in img:
                for i, caption in enumerate(img['tokenized_captions']):
                    img['tokenized_captions'][i] = caption.split()
            else:
                img['tokenized_captions'] = []
                for caption in img['captions']:
                    img['tokenized_captions'].append(tokenizer.pre_process(caption))

            img['tokens'] = []
            for i, caption in enumerate(img['tokenized_captions']):
                img['tokens'].append(token2index(caption, word_ids))

        random.seed(0)
        random.shuffle(formatted_val)
        val_img_num = int(len(formatted_val) * args.ratio)
        val_data = formatted_val[:val_img_num]
        test_data = formatted_val[val_img_num:]

    output_dataset = {}

    output_dataset['train'] = {'images': images, 'captions': captions, 'word_ids': word_ids}

    if args.exist_val:
        output_dataset['val'] = val_data
        output_dataset['test'] = test_data

    output_dict = word_ids

    # val and test data is not pre processed completely
    save_pickle(output_dataset, args.output_dataset_path)
    save_pickle(output_dict, args.output_dict_path)
