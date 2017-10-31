import json
import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description="convert machine translated json file to MSCOCO original formatted json file")
parser.add_argument('--input_mt', '-imt', type=str, default=os.path.join('..', '..', 'data', 'captions', 'original', 'MSCOCO_Chinese_translation', 'captions_train2014_cn_translation.json'),
                    help="input machine translated caption file")
parser.add_argument('--input_original_train_file', type=str, default=os.path.join('..', '..', 'data', 'captions', 'original', 'MSCOCO_captions_en', 'captions_train2014.json'),
                    help="input MSCOCO original formatted file")
parser.add_argument('--output_path', '-op', type=str, default=os.path.join('..', '..', 'data', 'captions', 'original', 'MSCOCO_Chinese_translation', 'captions_train2014_cn_translation_with_images.json'))

args = parser.parse_args()

with open(args.input_mt, 'r') as f:
    mt = json.load(f)

with open(args.input_original_train_file, 'r') as f:
    original = json.load(f)

mt_annot = mt['annotations']
images = original['images']

output = {'annotations': mt_annot, 'images': images}

with open(args.output_path, 'w') as f:
    json.dump(output, f, indent=4)
