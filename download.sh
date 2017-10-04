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
    echo -e 'Downloading MS COCO Datasets...\n'
    
    if [ ! -d data/images ]; then
        mkdir --parents data/images
    fi

    cd data/images

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
    echo -e 'Downloading MS COCO Captions...\n'

    if [ ! -d data/captions ]; then
        mkdir --parents data/captions/original data/captions/converted data/captions/processed
    fi

    cd data/captions/original

    #download official captions
    # annotations for test data is not offered
    curl -#O http://images.cocodataset.org/annotations/annotations_trainval2014.zip
    unzip annotations_trainval2014.zip
    rm annotations_trainval2014.zip


    #download STAIR captions
    curl -#O https://github.com/STAIR-Lab-CIT/STAIR-captions/raw/master/stair_captions_v1.1_train.json.tar.gz
    curl -#O https://github.com/STAIR-Lab-CIT/STAIR-captions/raw/master/stair_captions_v1.1_val.json.tar.gz

    tar xvzf STAIR-captions/stair_captions_v1.1_train.json.tar.gz
    tar xvzf STAIR-captions/stair_captions_v1.1_val.json.tar.gz


    #download Yahoo japan captions
    curl -#O https://github.com/yahoojapan/YJCaptions/raw/master/yjcaptions26k.zip
    unzip yjcaptions26k

fi
