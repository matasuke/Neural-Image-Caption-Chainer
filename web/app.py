import os
import sys
import flask
from flask import Flask, render_template, request, jsonify, Response
import numpy as np
import base64
import argparse
import cv2
import csv
from json import dumps
sys.path.append('../src')
from CaptionGenerator import CaptionGenerator

#model configuration
cnn_model_path = os.path.join('..', 'data', 'models', 'cnn', 'ResNet50.model')
rnn_model_path = os.path.join('..', 'data', 'models', 'rnn', 'STAIR_jp.model')
dict_path = os.path.join('..', 'data', 'vocab_dict', 'STAIR_jp.json')
beamsize = 3
depth_limit = 50
gpu = 0
first_word = '<S>'
hidden_dim = 512
mean = 'imagenet'

model = CaptionGenerator(rnn_model_path=rnn_model_path, cnn_model_path=cnn_model_path, dict_path=dict_path, cnn_model_type='ResNet', beamsize=beamsize, depth_limit=depth_limit, first_word=first_word, hidden_dim=hidden_dim, mean=mean)

app = Flask(__name__)

@app.route('/', methods=['GET'])
def generate_caption():
    if 'file' not in request.files:
        err = "File doesn't exist."
        return err

    img = request.files['file']
    bytes_img = base64.b64encode(img)
    decoded_img = bytes_img.decode('utf-8')

#@app.route('/api', method=['POST']):

#@app.route('/experiments'):


if __name__=='__main__':
    app.run()
