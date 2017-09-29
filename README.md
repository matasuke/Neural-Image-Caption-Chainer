# Neural Image Caption Chainer
- This is Neural Image Caption implementation by Chainer for Japanese caption.

## Datasets
[STAIR Captions](https://stair-lab-cit.github.io/STAIR-captions-web/)

## USAGE

### Prepare datasets.

1. the function below change mscoco dataset and annotations to the file that is easy to use.

`python preprocess_mscoco2converted.py`

2. this function split datasets and save them for easy usage. also tokenize them.
`python preprocess_captions.py`

