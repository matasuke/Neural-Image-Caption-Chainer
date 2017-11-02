#!bin/sh
python web/app.py \
    --rnn_model_jp_path ../data/models/rnn/STAIR_jp_256_Adam.model \
    --rnn_model_en_path ../data/models/rnn/MSCOCO_en_256_Adam.model \
    --rnn_model_ch_path ../data/models/rnn/MSCOCO_ch_mt_256_Adam.model \
    --cnn_model_path ../data/models/cnn/ResNet50.model \
    --dict_jp_path ../data/vocab_dict/dict_STAIR_jp_train.pkl \
    --dict_en_path ../data/vocab_dict/dict_MSCOCO_en_train.pkl \
    --dict_ch_path ../data/vocab_dict/dict_MSCOCO_ch_mt.pkl \
    --cnn_model_type ResNet \
    --gpu 0
