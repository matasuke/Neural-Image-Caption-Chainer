#!bin/sh

return_yes_or_no(){

    _ANSWER=

    while :
    do
        if [ "`echo -n`" = "-n" ]; then
            echo "$@\c"
        else
            echo -n "$@"
        fi
        read _ANSWER
        case "$_ANSWER" in
            [yY] | yes | YES | Yes) return 0 ;;
            [nN] | no | NO | No ) return 1 ;;
            * ) echo "type yes or no."
        esac
    done
}

echo 'Do you want to download pre-trained cnn models? (yes/no): '
cnn_models_download=`return_yes_or_no`
echo 'Do you want to download pre-trained rnn models? (yes/no): '
rnn_models_download=`return_yes_or_no`
echo 'Do you want to download images? (yes/no): '
images_download=`return_yes_or_no`
echo 'Do you want to download image features? (yes/no): '
feature_download=`return_yes_or_no`
echo 'Do you want to download captions? (yes/no): '
caption_download=`return_yes_or_no`


if $cnn_models_download ; then


    if [ ! -d data/models/cnn ]; then
        mkdir --parents data/models/cnn
        echo 'Downloading pre-trained cnn models'
    fi

    cd data/models/cnn
    if [ ! -f ResNet50.model ]; then
        wget https://www.dropbox.com/s/31mjg6hab2p2ei8/ResNet50.model
    fi

    cd ../../../
fi

if $rnn_models_download ; then

    if [ ! -d data/models/rnn ] ; then
        mkdir --parents data/models/rnn
        echo 'Downloading pre-trained cnn models'
    fi

    cd data/models/rnn
    #download trained models
    cd ../../../
fi

#downloading images

if $images_download ; then

    
    if [ ! -d data/images/original ]; then
        mkdir --parents data/images/original
        echo 'Downloading MS COCO Datasets...'
    fi

    cd data/images/original


    if [ ! -d train2014 ]; then
        curl -O http://images.cocodataset.org/zips/train2014.zip
        unzip train2014.zip
        rm train2014.zip
    fi

    if [ ! -d val2014 ]; then
        curl -O http://images.cocodataset.org/zips/val2014.zip
        unzip val2014.zip
        rm val2014.zip
    fi

    if [ ! -d test2014 ]; then
        curl -O http://images.cocodataset.org/zips/test2014.zip
        unzip test2014.zip
        rm test2014.zip
    fi

    if [ ! -d test2015 ]; then
        curl -O http://images.cocodataset.org/zips/test2015.zip
        unzip test2015.zip
        rm test2015.zip
    fi

    cd ../../../

fi

#download image features

if $feature_download; then

    if [ ! -d data/images/features ]; then
        mkdir --parents data/images/features
        echo 'Downloading MSCOCO Image features...'
    fi

    cd data/images/features

    if [ ! -d ResNet50 ]; then
        wget https://www.dropbox.com/s/ul0l5ps8ejhrwb0/ResNet50.tar.gz
        tar -zxvf ResNet50.tar.gz
        rm ResNet50.tar.gz
    fi

    cd ../../../
fi

#download caption data

if $caption_download; then


    if [ ! -d captions ]; then
        mkdir --parents data/captions/original data/captions/converted data/captions/processed data/captions/bleu
    fi

    cd data/captions/original

    #download official captions
    # annotations for test data is not offered
    if [ ! -d MSCOCO_captions_en ]; then
        
        echo 'Downloading MS COCO Captions...'
        curl -#O http://images.cocodataset.org/annotations/annotations_trainval2014.zip
        unzip annotations_trainval2014.zip
        rm annotations_trainval2014.zip
        mv annotations MSCOCO_captions_en

    fi

    #download STAIR captions
    if [ ! -d STAIR_captions ]; then
        mkdir STAIR_captions
    fi
    
    cd STAIR_captions

    if [ ! -f stair_captions_v1.1_train.json ]; then
        wget https://github.com/STAIR-Lab-CIT/STAIR-captions/raw/master/stair_captions_v1.1_train.json.tar.gz
        tar -zxvf stair_captions_v1.1_train.json.tar.gz
        rm stair_captions_v1.1_train.json.tar.gz
    fi

    if [ ! -f stair_captions_v1.1_val.json ]; then
        wget https://github.com/STAIR-Lab-CIT/STAIR-captions/raw/master/stair_captions_v1.1_val.json.tar.gz
        tar -zxvf stair_captions_v1.1_val.json.tar.gz
        rm stair_captions_v1.1_val.json.tar.gz
    fi

    cd ..


    #download Yahoo japan captions
    if [ ! -d Yahoo_captions ]; then
        mkdir Yahoo_captions
    fi

    cd Yahoo_captions

    if [ ! -f yjcaptions26k_clean.json ]; then
        wget https://github.com/yahoojapan/YJCaptions/raw/master/yjcaptions26k.zip
        unzip yjcaptions26k
        rm yjcaptions26k.zip
    fi

    cd ..

    #download machine translated chinese MACOCO captions
    if [ ! -d MSCOCO_Chinese_translation ]; then
        mkdir MSCOCO_Chinese_translation
    fi

    cd MSCOCO_Chinese_translation

    if [ ! -f captions_train2014_cn_translation.json ]; then
        wget https://github.com/apple2373/mt-mscoco/raw/master/captions_train2014_cn_translation.json.zip
        unzip captions_train2014_cn_translation.json.zip
        rm captions_train2014_cn_translation.json.zip
    fi

fi
