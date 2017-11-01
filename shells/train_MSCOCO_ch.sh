# train chinese caption
python train/train_model.py \
    --gpu 0 \
    --dataset ../data/captions/processed/dataset_MSCOCO_ch_mt.pkl \
    --img_feature_root ../data/images/features/ResNet50/ \
    --img_root ../data/images/original \
    --output_dir ../data/train_data/ \
    --preload \
    --epoch 100 \
    --batch_size 256 \
    --hidden_dim 512 \
    --img_feature_dim 2048 \
    --optimizer Adam \
    --dropout_ratio 0.5 \
    --n_layers 1 \
    --L2norm 1.0 \
