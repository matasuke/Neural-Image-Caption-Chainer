#!bin/bash

if [ ! -d models ]; then
    mkdir models
fi

cd models

if [ ! -d cnn ]; then
    mkdir cnn
fi

if [ ! -d rnn ]; then
    mkdir rnn
fi

cd ..

if [ ! -d images ]; then
    mkdir images
fi

cd images


#download caption data

if [ ! -d data/images -o ! -d data/captions ]; then
    mkdir --parents  data/images data/captions
fi

cd data/captions


wget https://github.com/STAIR-Lab-CIT/STAIR-captions/raw/master/stair_captions_v1.1_train.json.tar.gz
wget https://github.com/STAIR-Lab-CIT/STAIR-captions/raw/master/stair_captions_v1.1_val.json.tar.gz

tar xvzf STAIR-captions/stair_captions_v1.1_train.json.tar.gz
tar xvzf STAIR-captions/stair_captions_v1.1_val.json.tar.gz

rm *.tar.gz

wget https://github.com/yahoojapan/YJCaptions/raw/master/yjcaptions26k.zip
unzip yjcaptions26j.zip

rm yjcaptions26k.zip
