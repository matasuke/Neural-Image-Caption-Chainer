import argparse
import pickle
import json
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(description='create datasets from JSON')
parser.add_argument('input_train', type=str, 
                        help="input train JSON file path")

parser.add_argument('input_val', type=str,
                        help="input val JSON file path")

parser.add_argument('output', type=str,
                        help="output file path")

parser.add_argument('--split_ratio', '-s', type=float, default=0.4,
                        help="the ratio of split validation data and test data")

args = parser.parse_args()


#split ratio for validation data to vald data and test data
split_ratio = args.split_ratio

with open(args.input_train) as f:
    dataset_train = json.load(f)

with open(args.input_val) as f:
    dataset_val = json.load(f)


# associate words with numbers
word_ids = {
        '<S>': 0,
        '</S>': 1,
        '<UNK>': 2
}

unknown_word = 2
min_words_num = 5

list_sentences = {
        k: { n: [] for n in range(1, 91) } for k in ['train', 'val', 'test']
}
list_img_ids = {
        k: {n: [] for n in range(1, 91) } for k in ['train', 'val', 'test']
}


def words2ids(tokens):
    return [ word_ids[token] if token in word_ids else unknown_word for token in tokens ]


# don't add words that appear few times to the dict
word_counter = {}

# create vocabrary dictionary
for annotation in tqdm(dataset_train['annotations']):
    caption = annotation['tokenized_caption']
    tokens = caption.split()
    
    for token in tokens:
        if token in word_counter:
            word_counter[token] += 1
        else:
            word_counter[token] = 1

for word, num in word_counter.items():
    if num > min_words_num:
        word_ids[word] = len(word_ids)

#training data
for img in tqdm(dataset_train['annotations']):
    img_id = img['image_id']
    caption = img['tokenized_caption']
    tokens = caption.split()
    list_sentences['train'][len(tokens)].append(words2ids(tokens))
    list_img_ids['train'][len(tokens)].append(img_id)


#validation data and test data
np.random.shuffle(dataset_val['annotations'])
val_img_num = int(len(dataset_val['annotations']) * split_ratio)

for i, img in enumerate(tqdm(dataset_val['annotations'])):
    img_id = img['image_id']
    caption = img['tokenized_caption']
    tokens = caption.split()
    
    if i < val_img_num:
        list_sentences['val'][len(tokens)].append(words2ids(tokens))
        list_img_ids['val'][len(tokens)].append(img_id)
    else:
        list_sentences['test'][len(tokens)].append(words2ids(tokens))
        list_img_ids['test'][len(tokens)].append(img_id)


output_dataset = {}
output_dataset['sentences'] = {
        k: {n: np.array(sentences, dtype=np.int32) for n, sentences in v.items() } for k, v in list_sentences.items()
}
output_dataset['word_ids'] = word_ids
output_dataset['images'] = {
        k: {n: np.array(img_ids, dtype=np.int32) for n, img_ids in v.items() } for k, v in list_img_ids.items()
}

with open(args.output, 'wb') as f:
    pickle.dump(output_dataset, f, pickle.HIGHEST_PROTOCOL)
