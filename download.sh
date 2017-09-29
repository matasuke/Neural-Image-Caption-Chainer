#!bin/bash

return_yes_or_no(){
    while true;do
        
        read answer

        case $answer in
            yes)
                break
                ;;
            no)
                break
                ;;
            *)
                echo -e 'type yes or no\n'
                ;;
        esac
    done

    return $answer
}


echo 'Do you want to download pre-learned models(yes/no)?'
model_download=return_yes_or_no
echo 'Do you want to download images(yes/no)?'
image_download=return_yes_or_no
echo 'Do you want to download captions(yes/no)?'
caption_download=return_yes_or_no


if [ $model_download == 'yes' ]; then
    if [ ! -d models/cnn -o ! -d models/rnn ]; then
        mkdir --parents models/cnn models/rnn
    fi
fi


#downloading images

if [ $image_download == 'yes' ]; then
    echo -e 'Downloading MS COCO Datasets...\n'
    
    if [ ! -d data/images ]; then
        mkdir --parents data/images
    fi

    cd data/images

    if [ ! -d train2014 ]; then
        curl -O http://images.cocodataset.org/zips/train2014.zip
        unzip -l train2014.zip
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


#download caption data

if [ $caption_download == 'yes' ]; then
    echo -e 'Downloading MS COCO Captions...\n'

    if [ ! -d data/captions/original ]; then
        mkdir --parents data/captions/original data/captions/converted
    fi

    cd data/captions

    #download official captions
    wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
    unzip -l annotations_trainval2014.zip

    rm annotations_trainval2014.zip

    #download STAIR captions
    wget https://github.com/STAIR-Lab-CIT/STAIR-captions/raw/master/stair_captions_v1.1_train.json.tar.gz
    wget https://github.com/STAIR-Lab-CIT/STAIR-captions/raw/master/stair_captions_v1.1_val.json.tar.gz

    tar xvzf STAIR-captions/stair_captions_v1.1_train.json.tar.gz
    tar xvzf STAIR-captions/stair_captions_v1.1_val.json.tar.gz

    rm *.tar.gz

    #download Yahoo japan captions
    wget https://github.com/yahoojapan/YJCaptions/raw/master/yjcaptions26k.zip
    unzip yjcaptions26j.zip

    rm yjcaptions26k.zip
fi
