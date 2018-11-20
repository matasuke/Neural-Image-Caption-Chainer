import json
from pathlib import Path
import argparse

DEFAULT_INPUT_PATH = Path('data/captions/original/MSCOCO_Chinese_translation/captions_train2014_cn_translation.json')
DEFAULT_TRAIN_FILE_PATH = Path('data/captions/original/MSCOCO_captions_en/captions_train2014.json')
DEFAULT_OUTPUT_PATH = \
    Path('data/captions/original/MSCOCO_Chinese_translation/captions_train2014_cn_translation_with_images.json')


parser = argparse.ArgumentParser(
    description="convert machine translated json file to MSCOCO original formatted json file"
)
parser.add_argument(
    '--input_mt', '-imt', type=str,
    default=DEFAULT_INPUT_PATH.as_posix(),
    help="input machine translated caption file"
)
parser.add_argument(
    '--input_original_train_file', type=str,
    default=DEFAULT_TRAIN_FILE_PATH.as_posix(),
    help="input MSCOCO original formatted file"
)
parser.add_argument(
    '--output_path', '-op', type=str,
    default=DEFAULT_OUTPUT_PATH.as_posix(),
)

args = parser.parse_args()

with open(args.input_mt) as f:
    mt = json.load(f)

with open(args.input_original_train_file, 'r') as f:
    original = json.load(f)

mt_annot = mt['annotations']
images = original['images']

output = {'annotations': mt_annot, 'images': images}

with open(args.output_path, 'w') as f:
    json.dump(output, f, indent=4)
