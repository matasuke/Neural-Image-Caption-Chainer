import os
import sys
import argparse
import flask
from flask import Flask, render_template, request, jsonify, Response
from werkzeug import secure_filename
import numpy as np
import base64
from chainer import serializers
from json import dumps
sys.path.append('../src')
sys.path.append('../src/CNN')
from CaptionGenerator import CaptionGenerator
sys.path.append('..')
import WEB_ENV
from PIL import Image

def model_configuration(args):
    
    #model configuration
    cnn_model_path = args.cnn_model_path
    cnn_model_type = args.cnn_model_type
    rnn_model_jp_path = args.rnn_model_jp_path
    rnn_model_en_path = args.rnn_model_en_path
    dict_jp_path = args.dict_jp_path
    dict_en_path = args.dict_en_path
    beamsize = args.beamsize
    depth_limit = args.depth_limit
    gpu = args.gpu
    first_word = args.first_word
    hidden_dim = args.hidden_dim
    mean = args.mean

    conf_dict = {'cnn_model_path': cnn_model_path, 'cnn_model_type': cnn_model_type, 'rnn_model_jp_path': rnn_model_jp_path, 'rnn_model_en_path': rnn_model_en_path, 'dict_jp_path': dict_jp_path, 'dict_en_path': dict_en_path, 'beamsize': beamsize, 'depth_limit': depth_limit, 'gpu': gpu, 'first_word': first_word, 'hidden_dim': hidden_dim, 'mean': mean}

    return conf_dict

def allowed_file(filename):
    if '.' in filename and \
            filename.rsplit('.')[1].lower() in WEB_ENV.ALLOWED_EXTENSIONS:
        return True
    else:
        return False

def connect_tokens_jp(tokens):
    return ''.join(tokens)

def connect_tokens_en(tokens):
    return ' '.join(tokens)

def parse_captions_jp(captions):
    output = []
    for i, caption in enumerate(captions):
        output.append({'No': i, 'caption': connect_tokens_jp(caption['sentence'][1:-1]), 'tokens': caption['sentence'], 'log': caption['log_likelihood'], 'num_tokens': len(caption['sentence'])})
    
    return output

def parse_captions_en(captions):
    output = []
    for i, caption in enumerate(captions):
        output.append({'No': i, 'caption': connect_tokens_en(caption['sentence'][1:-1]), 'tokens': caption['sentence'], 'log': caption['log_likelihood'], 'num_tokens': len(caption['sentence'])})
    
    return output

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = WEB_ENV.UPLOAD_FOLDER

@app.route('/', methods=['GET'])
def generate_caption():
    title = 'Neural Image Caption Experiment Platform'
    conf = model_configuration(args)
    return render_template('index.html', title = title, configuration = conf)

@app.route('/api', methods=['POST'])
def return_captions():

    if request.method == 'POST':
        if request.headers['Content-Type'] == 'multipart/form-data':
            if 'file' not in request.files:
                return None
            else:
                print("test")
                img = request.files['file']

        img = request.files['file']
    if img and allowed_file(img.filename):
        filename = secure_filename(img.filename)
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        img.save(img_path)

    output = {}
    jp_captions = jp_model.generate(img_path)
    en_captions = en_model.generate(img_path)
    
    output.update({'jp': parse_captions_jp(jp_captions)})
    output.update({'en': parse_captions_en(en_captions)})

    return jsonify(output)

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--rnn_model_jp_path', '-rjm', type=str, default=os.path.join('..', 'data', 'models', 'rnn', 'STAIR_jp_256_Adam.model'),
                        help="RNN japanese model path")
    parser.add_argument('--rnn_model_en_path', '-rem', type=str, default=os.path.join('..', 'data', 'models', 'rnn', 'MSCOCO_en_256_Adam.model'),
                        help="RNN english model path")
    parser.add_argument('--cnn_model_path', '-cm', type=str, default=os.path.join('..', 'data', 'models', 'cnn', 'ResNet50.model'),
                        help="CNN model path")
    parser.add_argument('--dict_jp_path', '-jd', type=str, default=os.path.join('..', 'data', 'captions', 'processed', 'dataset_STAIR_jp.pkl'),
                        help="Japanese Dictionary path")
    parser.add_argument('--dict_en_path', '-ed', type=str, default=os.path.join('..', 'data', 'captions', 'processed', 'dataset_MSCOCO_en.pkl'),
                        help="English Dictionary path")
    parser.add_argument('--cnn_model_type', '-ct', type=str, choices = ['ResNet', 'VGG16', 'AlexNet'], default="ResNet")
    parser.add_argument('--beamsize', '-b', type=str, default=3,
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

    jp_model = CaptionGenerator(
            rnn_model_path = args.rnn_model_jp_path,
            cnn_model_path = args.cnn_model_path,
            dict_path = args.dict_jp_path,
            cnn_model_type = args.cnn_model_type,
            beamsize = args.beamsize,
            depth_limit = args.depth_limit,
            gpu_id = args.gpu,
            first_word = args.first_word,
            hidden_dim = args.hidden_dim,
            mean = args.mean)
    
    en_model = CaptionGenerator(
            rnn_model_path = args.rnn_model_en_path,
            cnn_model_path = args.cnn_model_path,
            dict_path = args.dict_en_path,
            cnn_model_type = args.cnn_model_type,
            beamsize = args.beamsize,
            depth_limit = args.depth_limit,
            gpu_id = args.gpu,
            first_word = args.first_word,
            hidden_dim = args.hidden_dim,
            mean = args.mean)

    configurations = model_configuration(args)
    
    app.run(debug=True)
