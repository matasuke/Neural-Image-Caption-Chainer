import json
import pickle
import os
import argparse
from tqdm import tqdm

'''
This script converts MS COCO Json file to formatted one that is converted to pickle format

USAGE:

python preprocess_mscoco2convert.py \
--input_train ../data/captions/original/stair_captions_v1.1_train.json \
--input_val ../data/captions/original/stair_captions_v1.1_val.json \
--out_dir ../data/captions/converted
'''


def read_mscoco(json_file):
    with open(json_file) as f:
        dataset = json.load(f)
    
    annots = dataset['annotations']
    imgs = dataset['images']

    return imgs, annots

def save_mscoco(out_data, pickle_file):
    with open(pickle_file, 'wb') as f:
        pickle.dump(out_data, f, pickle.HIGHEST_PROTOCOL)

def make_groups(annots):
    itoa = {}
    
    for a in tqdm(annots):
        img_id = a['image_id']
        if not img_id in itoa:
            itoa[img_id] = []
        itoa[img_id].append(a)

    return itoa

def create_converted(itoa, imgs):
    has_token = False
    out_data = []

    for i, img in enumerate(tqdm(imgs)):
        img_id = img['id']

        data_type = 'train2014' if 'train' in img['file_name'] else 'val2014'

        pairs = {}
        pairs['file_path'] = os.path.join(data_type, img['file_name'])
        pairs['id'] = img_id
    
        sentences = []
        annots = itoa[img_id]

        if 'tokenized_caption' in annots[0]:
            token = True
            tokenized = []

        for a in annots:
            sentences.append(a['caption'])
            if token:
                tokenized.append(a['tokenized_caption'])
       
        pairs['captions'] = sentences
        if token:
            pairs['tokenized_captions'] = tokenized

        out_data.append(pairs)

    return out_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert MS COCO formatted json to originally formatted ones')
    parser.add_argument('--input_train', '-it', type=str, default="../data/captions/original/stair_captions_v1.1_train.json",
                        help="input train JSON file path")
    parser.add_argument('--input_val', '-iv', type=str, default="../data/captions/original/stair_captions_v1.1_val.json",
                        help="input val JSON file path")
    parser.add_argument('--out_dir', '-od', type=str, default="../data/captions/converted",
                        help="output dir path")
    parser.add_argument('--output_train', '-ot', type=str, default="formatted_json_train_jp.pkl", 
                        help="output file name for train data")
    parser.add_argument('--output_val', '-ov', type=str, default='formatted_json_val_jp.pkl',
                        help="output file name for val data")
    args = parser.parse_args()

    imgs_t, annots_t = read_mscoco(args.input_train)
    imgs_v, annots_v = read_mscoco(args.input_val)

    itoa_t = make_groups(annots_t)
    itoa_v = make_groups(annots_v)

    out_data_t = create_converted(itoa_t, imgs_t)
    out_data_v = create_converted(itoa_v, imgs_v)

    out_path_t = os.path.join(args.out_dir  , args.output_train)
    out_path_v = os.path.join(args.out_dir  , args.output_val)
    
    print('Saveing pkl file...')
    
    save_mscoco(out_data_t, out_path_t)
    save_mscoco(out_data_v, out_path_v)
