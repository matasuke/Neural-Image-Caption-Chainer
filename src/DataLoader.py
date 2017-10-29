import numpy as np
import os
from img_proc import Img_proc


class DataLoader:
    def __init__(self, dataset=os.path.join('..', 'data', 'captions', 'processed', 'dataset_STAIR_jp.pkl'), img_feature_root=os.path.join('..', 'data', 'images', 'features', 'ResNet50',), img_root=os.path.join('..', 'data', 'images', 'original'), img_mean="imagenet", preload_features=False, exist_test = True):
        self.train = dataset['train']
        if exist_test:
            self.val = dataset['val']
            self.test = dataset['test']
        self.preload_features = False
        self.img_proc = Img_proc(mean_type=img_mean)
        self.captions = self.train['captions']
        self.token2index = self.train['word_ids']
        self.index2token = { v: k for k, v in self.token2index.items() }
        self.num_captions = len(self.captions)
        self.images = self.train['images']
        self.num_images = len(self.images)
        self.cap2img = { caption['caption_idx']:caption['img_idx'] for caption in self.train['captions'] }
        self.img_feature_root = img_feature_root
        self.img_root = img_root
        self.random_indices = np.random.permutation(len(self.captions))
        self.index_counter = 0
        self.epoch = 1
        self.preload_features = preload_features
        if self.preload_features:
            self.img_features_train = np.array([np.load( '{0}.npz'.format(os.path.join(self.img_feature_root, os.path.splitext(image['file_path'])[0])))['arr_0'] for image in self.images ])
            self.img_features_val = np.array([np.load( '{0}.npz'.format(os.path.join(self.img_feature_root, os.path.splitext(val['file_path'])[0])))['arr_0'] for val in self.val ])   
    
    def get_batch(self, batch_size=248, raw_img = False, raw_captions=False):
        batch_caption_indices = self.random_indices[self.index_counter: self.index_counter + batch_size]
        self.index_counter += batch_size
        
        if self.index_counter > self.num_captions - batch_size:
            self.epoch += 1
            self.shuffle_data()
            self.index_counter = 0

        if raw_img:
            batch_images = np.array([ self.img_proc.load_img(os.path.join(self.img_root, self.images[self.cap2img[i]]['file_path']), expand_dim = False) for i in batch_caption_indices ])
        else:
            if self.preload_features:
                batch_images = self.img_features_train[[self.cap2img[i] for i in batch_caption_indices ]]
            else:
                batch_images = np.array([ np.load( '{0}.npz'.format(os.path.join(self.img_feature_root, os.path.splitext(self.images[self.cap2img[i]]['file_path'])[0] )))['arr_0'] for i in batch_caption_indices ])
        
        if raw_captions:
            batch_word_indices = [ self.captions[i]['caption'] for i in batch_caption_indices ]
        else:
            batch_word_indices = [ np.array(self.captions[i]['caption'], dtype=np.int32) for i in batch_caption_indices]

        return batch_images, batch_word_indices

    def get_batch_raw(self, batch_size=248, raw_img=False):
        batch_caption_indices = self.random_indices[self.index_counter: self.index_counter + batch_size]
        self.index_counter += batch_size
        
        if self.index_counter > self.num_captions - batch_size:
            self.epoch += 1
            self.shuffle_data()
            self.index_counter = 0

        batch_images_path = [ self.images[self.cap2img[i]]['file_path' ] for i in batch_caption_indices ]

        batch_word_indices = [ self.captions[i]['caption'] for i in batch_caption_indices ]

        for i, cap in enumerate(batch_word_indices):
            for j, c in enumerate(cap):
                batch_word_indices[i][j] = self.index2token[c]


        return batch_images_path, batch_word_indices

    def shuffle_data(self):
        self.random_indices = np.random.permutation(len(self.captions))

    def get_val(self, raw_img = False, raw_captions=False):
        
        if raw_img:
            val_images = np.array([ self.img_proc.load_img(os.path.join(self.img_root, i['file_path']), expand_dim = False) for i in self.val ])
        else:
            if self.preload_features:
                val_images = self.img_features_val
            else:
                val_images = np.array([ np.load( '{0}.npz'.format(os.path.join(self.img_feature_root, os.path.splitext(i['file_path'])[0] )))['arr_0'] for i in self.val ])

        if raw_captions:
            val_captions = []
            for i in self.val:
                val_captions.append([j for j in i['tokens'] ])
        else:
            val_captions = []
            for i in self.val:
                val_captions.append([ np.array(j, dtype=np.int32) for j in i['tokens'] ])
                
        return val_images, val_captions

    @property
    def caption_size(self):
        return self.num_captions

    @property
    def img_size(self):
        return self.num_images

    @property
    def now_epoch(self):
        return self.epoch

    @property
    def is_new_epoch(self):
        pass 


if __name__ == "__main__":
    import pickle
    with open(os.path.join('..', 'data', 'captions', 'processed', 'dataset_STAIR_jp.pkl'), 'rb') as f:
        data = pickle.load(f)

    dataset = DataLoader(data, img_feature_root=os.path.join('..', 'data', 'images', 'features', 'ResNet50'), img_root=os.path.join('..', 'data', 'images', 'original'), preload_features=True, exist_test=True)
    
    batch_images_path, batch_word_indices = dataset.get_batch_raw(10)
    for path, words in zip(batch_images_path, batch_word_indices):
        print('img:', path)
        print('words', words)
    
    val_images, val_captions = dataset.get_val(raw_img=False, raw_captions=False)
    
    for img, words in zip(val_images, val_captions):
        print('img:', img)
        print('words', words)

'''
    dataset = DataLoader(data, img_feature_root=os.path.join('..', 'data', 'images', 'features', 'ResNet50'), preload_features=True)
    batch_images_path, batch_word_indices = dataset.get_batch_raw(10)
    for path, words in zip(batch_images_path, batch_word_indices):
        print('img:', path)
        print('words', words)

    batch_images, batch_word_indices = dataset.get_batch(10, raw_img=True, raw_captions=True)
    for img, words in zip(batch_images, batch_word_indices):
        print('img:', img)
        print('words', words)

    batch_images, batch_word_indices = dataset.get_batch(10, raw_img=False, raw_captions=True)
    for img, words in zip(batch_images, batch_word_indices):
        print('img:', img)
        print('words', words)
    
    batch_images, batch_word_indices = dataset.get_batch(10, raw_img=True, raw_captions=False)
    for img, words in zip(batch_images, batch_word_indices):
        print('img:', img)
        print('words', words)
    
    dataset = DataLoader(data, img_feature_root=os.path.join('..', 'data', 'images', 'features', 'ResNet50'), img_root=os.path.join('..', 'data', 'images', 'original'))
    batch_images, batch_word_indices = dataset.get_batch(10, raw_img=True)
    
    for img, words in zip(batch_images, batch_word_indices):
        print('img:', img)
        print('words', words)

    batch_images, batch_word_indices = dataset.get_batch(10)
    
    for img, words in zip(batch_images, batch_word_indices):
        print('img:', img)
        print('words', words)
'''

