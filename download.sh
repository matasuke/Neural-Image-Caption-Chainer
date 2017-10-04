#!bin/bash

return_yes_or_no(){
    while true;do
        
        read answer

        case $answer in
            [Yy]* )
                return 0
                ;;
            [Nn]* )
                return 1
                ;;
            *)
                echo -e 'type yes or no\n'
                ;;
        esac
    done
}


echo 'Do you want to download pre-learned models(yes/no)?'
model_download=`return_yes_or_no`
echo 'Do you want to download images(yes/no)?'
image_download=`return_yes_or_no`
echo 'Do you want to download captions(yes/no)?'
caption_download=`return_yes_or_no`

echo $model_download

if [ ! $model_download ]; then
    if [ ! -d models/cnn -o ! -d models/rnn ]; then
        mkdir --parents models/cnn models/rnn
    fi
fi


#downloading images

if [ ! $image_download ]; then
    echo 'Downloading MS COCO Datasets...\n'
    
    if [ ! -d data/images/original ]; then
        mkdir --parents data/images/original
    fi

    cd data/images/original

    if [ ! -d train2014 ]; then
        curl -O http://images.cocodataset.org/zips/train2014.zip
        unzip train2014.zip
        rm train2014.zip
    fi

    if [ ! -d val2014 ]; then
        curl -O http://images.cocodataset.org/zips/val2014.zip
        unzip -l val2014.zip
        rm val2014.zip
    fi

    if [ ! -d test2014 ]; then
        curl -O http://images.cocodataset.org/zips/test2014.zip
        unzip -l test2014.zip
        rm test2014.zip
    fi

    if [ ! -d test2015 ]; then
        curl -O http://images.cocodataset.org/zips/test2015.zip
        unzip -l test2015.zip
        rm test2015.zip
    fi
fi


cd ../..

#download caption data

if [ ! $caption_download ]; then
    echo 'Downloading MS COCO Captions...\n'

    if [ ! -d captions ]; then
        mkdir --parents data/captions/original data/captions/converted data/captions/processed
    fi

    cd captions/original

    #download official captions
    # annotations for test data is not offered
    if [ ! -d annotations ]; then
        curl -#O http://images.cocodataset.org/annotations/annotations_trainval2014.zip
        unzip annotations_trainval2014.zip
        rm annotations_trainval2014.zip
        mv annotations mscoco_official_annotations_en
    fi


    #download STAIR captions
    if [ ! -d STAIR_Captions ]; then
        mkdir STAIR_Captions
    fi

    cd STAIR_Captions

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

fi
