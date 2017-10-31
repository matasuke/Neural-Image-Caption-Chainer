# prepare Japanese data
echo 'Process Japanese captions'
python ../src/DataPreparation/preprocess_mscoco2converted.py --input_train ../data/captions/original/STAIR_captions/stair_captions_v1.1_train.json --exist_val --input_val ../data/captions/original/STAIR_captions/stair_captions_v1.1_val.json --output_dir ../data/captions/converted/ --output_train formatted_json_train_jp.pkl --output_val formatted_json_val_jp.pkl

python ../src/DataPreparation/preprocess_captions.py --input_train ../data/captions/converted/formatted_json_train_jp.pkl --exist_val --input_val ../data/captions/converted/formatted_json_val_jp.pkl --output_dataset_path ../data/captions/processed/dataset_STAIR_jp.pkl --output_dict_path ../data/vocab_dict/dict_STAIR_jp_train.pkl --lang jp --off 5 --lower --period --ratio 0.5

# prepare English data
echo 'Process English captions'
python ../src/DataPreparation/preprocess_mscoco2converted.py --input_train ../data/captions/original/MSCOCO_captions_en/captions_train2014.json --exist_val --input_val ../data/captions/original/MSCOCO_captions_en/captions_val2014.json --output_dir ../data/captions/converted/ --output_train formatted_json_train_en.pkl --output_val formatted_json_val_en.pkl

python ../src/DataPreparation/preprocess_captions.py --input_train ../data/captions/converted/formatted_json_train_en.pkl --exist_val --input_val ../data/captions/converted/formatted_json_val_en.pkl --output_dataset_path ../data/captions/processed/dataset_MSCOCO_en.pkl --output_dict_path ../data/vocab_dict/dict_MSCOCO_en_train.pkl --lang en --off 5 --lower --period --ratio 0.5

# prepare Chinese data
echo 'Process Chinese captions'
python ../src/DataPreparation/convert_mt2mscoco.py --input_mt ../data/captions/original/MSCOCO_Chinese_translation/captions_train2014_cn_translation.json --input_original_train_file ../data/captions/original/MSCOCO_captions_en/captions_train2014.json --output_path ../data/captions/original/MSCOCO_Chinese_translation/captions_train2014_cn_translation_with_images.json

python ../src/DataPreparation/preprocess_mscoco2converted.py --input_train ../data/captions/original/MSCOCO_Chinese_translation/captions_train2014_cn_translation_with_images.json --output_dir ../data/captions/converted/ --output_train formatted_json_train_ch_mt.pkl

python ../src/DataPreparation/preprocess_captions.py --input_train ../data/captions/converted/formatted_json_train_ch_mt.pkl --output_dataset_path ../data/captions/processed/dataset_MSCOCO_ch_mt.pkl --output_dict_path ../data/vocab_dict/dict_MSCOCO_ch_mt_train.pkl --lang ch --off 5 --lower --period --ratio 0.5

