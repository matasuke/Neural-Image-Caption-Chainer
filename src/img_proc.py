import os
import numpy as np
import cv2


class Img_proc(object):
    def __init__(self, mean_type = None):
        
        if mean_type == None:
            self.mean = np.zeros([3, 1, 1])
        
        elif mean_type == 'imagenet':
            mean = np.ndarray([3, 224, 224], dtype=np.float32)
            mean[0] = 103.939
            mean[1] = 116.779
            mean[2] = 123.68
            self.mean = mean
        
        elif len(mean_type) == 3:
            mean = np.ndarray([3, 244, 244], dtype=np.float32)
            mean[0] = mean_type[0]
            mean[1] = mean_type[1]
            mean[2] = mean_type[2]
            self.mean = mean

    def load_img(self, img_path, img_h = 224, img_w = 224, resize = True, expand_dim = True):
        img = cv2.imread(img_path).astype(np.float32)

        if resize:
            size = (img_h, img_w)
            img = cv2.resize(img, size)
            img = img.transpose(2, 0, 1)
        else:
            size = (img.shape[0], img.shape[1])
            img = img.transpose(2, 0, 1)
        
        img -= self.mean

        if expand_dim:
            img = np.expand_dims(img, axis=0)

        return img

    def save_img(self, np_array, save_path):
        img = np_array.transpose(1, 2, 0)
        cv2.imwrite(save_path, img)
