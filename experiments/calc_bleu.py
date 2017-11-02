import os
import sys
import argparse
sys.path.append('../src')
from CaptionGenerator import CaptionGenerator

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_feature_dir', '-id', type=str, default=os.path.join('..', 'data', 'images', 'features', 'ResNet50', 'val2014'),
                        help="image feature dir")
    parser.add_argument('--val_path', '-vp', type=str, default=os.path.join('..', 'data', 'captions', 'processed', 'dataset_STAIR_jp.pkl'),
                        help="processed validation caption path")
    parser.add_argument('--rnn_model_path', '-rmp', type=str, default=os.path.join('..', 'data', 'models', 'rnn', 'STAIR_jp_256_Adam.model'),
                        help="RNN model path")
    parser.add_argument('--cnn_model_path', '-cmp', type=str, default=os.path.join('..', 'data', 'models', 'cnn', 'ResNet50.model'),
                        help="CNN model path")
    parser.add_argument('--dict_path', '-dp', type=str, default=os.path.join('..', 'data', 'vocab_dict', 'dict_STAIR_jp_train.pkl'),
                        help="Dictionary path")
    parser.add_argument('--cnn_model_type', '-ct', type=str, default='ResNet', choices = ['ResNet', 'VGG16', 'AlexNet'],
                        help="CNN model type")
    parser.add_argument('--output_path', '-op', type=str, defalut=os.path.join('.', 'bleu_score.txt'),
                        help="output path")
    parser.add_argument('--beamsize', '-b', type=str, default=3,
                        help="Beamsize")
    parser.add_argument('--depth_limit', '-dl', type=str, default=50,
                        help="max limit of generation tokens when constructing captions")
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help="set GPU ID(negative value means using CPU)")
    parser.add_argument('--first_word', '-fw', type=str, default='<S>',
                        help="set first word")
    parser.add_argument('--hidden_dim', '-hd', type=int, default=512,
                        help="dimension of hidden layers")
    parser.add_argument('--mean', '-m', type=str, choices=['imagenet'], default="imagenet",
                        help="method to preprocess images")
    args = parser.parse_args()
   
    caption_generator = CaptionGenerator(
            rnn_model_path = args.rnn_model_path,
            cnn_model_path = args.cnn_model_path,
            dict_path = args.dict_path,
            cnn_model_type = args.cnn_model_type,
            beamsize = args.beamsize,
            depth_limit = args.depth_limit,
            gpu_id = args.gpu_id,
            first_word = args.first_word,
            hidden_dim = args.hidden_dim,
            mean = args.mean)

    
    caption_generator.generate_from_img_feature()
