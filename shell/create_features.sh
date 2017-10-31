# feature extraction by resnet50
python src/CNN/resnet/ResNet_feature_extractor.py --input_dir ../../../data/images/original/train2014 --output_dir ../../../data/images/features/ResNet50/train2014 --model ../../../data/models/cnn/ResNet50.model --gpu 0

python src/CNN/resnet/ResNet_feature_extractor.py --input_dir ../../../data/images/original/val2014 --output_dir ../../../data/images/features/ResNet50/val2014 --model ../../../data/models/cnn/ResNet50.model --gpu 0
