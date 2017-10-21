# Neural Image Caption Chainer
This repository is about Neural Image Caption implementation by Chainer including Japanese, English and Chinese and so on

### Requirements
Chainer v3

## Datasets

### Images
- [MSCOCO Images](http://cocodataset.org/)

### Captions
- [MSCOCO Annotations](http://cocodataset.org/#download)
- [STAIR Captions](https://stair-lab-cit.github.io/STAIR-captions-web/)

## USAGE

### Prepare datasets.

1. the function below change mscoco dataset and annotations to the file that is easy to use.

`python preprocess_mscoco2converted.py`

2. this function split datasets and save them for easy usage. also tokenize them.
`python preprocess_captions.py`

## Language Availability
- prepared languages in this repository are below this
    + English
    + Japanese
- going to prepare these languages
    + Chinese
    + Korean
    + Cantonese
    + Russian
    + French
    + Spanish

## generate captions using prepared model

## train MSCOCO data

## train your own data


## Experiments

## Next to do
-  prepare multi language processer.
-  prepare datas of multi languages using google cloud platfor api.
-  prepare environments for experiment of generating captions
-  implement caption model using Bokete(create humor caption model)
-  implement web api

## Citations
