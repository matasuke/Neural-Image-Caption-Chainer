import numpy as np
import cv2 as cv


class Img_proc(object):
    def __init__(self, mean_type = None, img_w = 224, img_h = 224):
        
        self.img_w = img_w
        self.img_h = img_h
        self.size = (self.img_h, self.img_w)

        if mean_type == None:
            self.mean = np.zeros((3, 1, 1))
        
        elif mean_type == 'imagenet':
            mean = np.ndarray((3, self.img_h, self.img_w), dtype=np.float32)
            mean[0] = 103.939
            mean[1] = 116.779
            mean[2] = 123.68
            self.mean = mean
        
        elif len(mean_type) == 3:
            mean = np.ndarray((3, 244, 244), dtype=np.float32)
            mean[0] = mean_type[0]
            mean[1] = mean_type[1]
            mean[2] = mean_type[2]
            self.mean = mean

    def load_img(self, img_path, expand_dim = True):
        img = cv2.imread(img_path)
        resized = cv2.resize(img, self.size).transpose(2, 1, 0)
        resized -= self.mean
        
        if expand_dim:
            return resized.reshape(1, 3, self.img_h, self.img_w)
        else:
            return resized
