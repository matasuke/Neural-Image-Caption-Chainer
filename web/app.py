import os
import sys
import argparse
import flask
from flask import Flask, render_template, request, jsonify, Response, send_from_directory
import numpy as np
import base64
import argparse
import cv2
import csv
import WEB_ENV
from chainer import serializers
from json import dumps
sys.path.append('../src')
from CaptionGenerator import CaptionGenerator
sys.path.append('..')

def model_configuration(args):
    
    #model configuration
    cnn_model_path = args.cnn_model_path
    cnn_model_type = args.cnn_model_type
    rnn_model_path = args.rnn_model_path
    dict_path = args.dict_path
    beamsize = args.beamsize
    depth_limit = args.depth_limit
    gpu = args.gpu
    first_word = args.first_word
    hidden_dim = args.hidden_dim
    mean = args.mean

    conf_dict = {'cnn_model_path': cnn_model_path, 'cnn_model_type': cnn_model_type, 'rnn_model_path': rnn_model_path, 'dict_path': dict_path, 'beamsize': beamsize, 'depth_limit': depth_limit, 'gpu': gpu, 'first_word': first_word, 'hidden_dim': hidden_dim, 'mean': mean}

    return conf_dict

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = WEB_ENV.UPLOAD_FOLDER

@app.route('/', methods=['GET'])
def generate_caption():
    pass

@app.route('/api', method=['POST'])
def return_captions():

    if request.method == 'POST':
        if request.headers['Content-Type'] == 'images/jpeg':
            if 'file' not in request.files:
                return 'error'
            else:
                img = request.files['file']

    img_path = 
    cv2.imwrite(img_path, img)
    
    output = []
    captions = model.generate(img_path)
    
    for i, caption in enumerate(captions):
        output.append({'No': i, 'Caption': caption['sentence'], 'log': caption['log_likelihood']})

    return jsonify(output)

#@app.route('/experiments'):


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_lrgument('--rnn_model_path', '-rm', type=str, default=os.path.join('..', 'data', 'models', 'rnn', 'dataset_STAIR_jp_256_Adam.model'),
                        help="RNN model path")
    parser.add_argument('--cnn_model_path', '-cm', type=str, default=os.path.join('..', 'data', 'models', 'cnn', 'ResNet50.model'),
                        help="CNN model path")
    parser.add_argument('--dict_path', '-d', type=str, default=os.path.join('..', 'data', 'captions', 'processed', 'dataset_STAIR_jp.pkl'),
                        help="Dictionary path")
    parser.add_argument('--cnn_model_type', '-ct', type=str, choices = ['AdaDelta', 'AdaGrad', 'Adam', 'MomentumSGD', 'NesterovAG', 'RMSprop', 'RMSpropGraves', 'SGD', 'SMORMS3'])
    parser.add_argument('--beamsize', '-b', type=str, default=5,
                        help="beamsize")
    parser.add_argument('--depth_limit', '-dl', type=int, default=50,
                        help="max limit of generating tokens when constructing captions")
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help="set GPU ID(negative value means using CPU)")
    parser.add_argument('--first_word', '-fw', type=str, default="<S>",
                        help="set first word")
    parser.add_argument('--hidden_dim', '-hd', type=int, default=512,
                        help="dimension of hidden layers")
    parser.add_argument('--mean', '-m', type=str, choices=['imagenet'], default="imagenet",
                        help="method to preprocess images")
    args = parser.parse_args()

    model = CaptionGenerator(
            rnn_model_path = args.rnn_model_path,
            cnn_model_path = args.cnn_model_path,
            dict_path = args.dict_path,
            cnn_model_type = args.cnn_model_type,
            beamsize = args.beamsize,
            depth_limit = args.depth_limit,
            gpu_id = args.gpu,
            first_wird = args.first_word,
            hidden_dim = args.hidden_dim,
            mean = args.mean)
    

    configurations = model_configuration(args)
    app.run()
