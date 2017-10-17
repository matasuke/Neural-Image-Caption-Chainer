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

## Next to do
- [] prepare multi language processer.
- [] prepare datas of multi languages.
- [] use google cloud platform api for machine translation
- [] prepare environments for experiment of generating captions
- [] fix codes for chainer v3
- [] implement caption model using Bokete(create humor caption model)
- [] implement web api
