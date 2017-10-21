import numpy as np
import os
from img_proc import Img_proc


class DataLoader:
    def __init__(self, dataset, img_feature_root="../data/images/features/ResNet50/", img_root='../data/images/original/', img_mean="imagenet", raw_captions=False, preload_features=False):
        self.raw_captions = False
        self.preload_features = False
        self.img_proc = Img_proc(mean_type=img_mean)
        self.captions = dataset['captions']
        self.num_captions = len(self.captions)
        self.images = dataset['images']
        self.num_images = len(self.images)
        self.cap2img = { caption['caption_idx']:caption['img_idx'] for caption in dataset['captions'] }
        self.img_feature_root = img_feature_root
        self.img_root = os.path.join(img_root, '')
        self.random_indices = np.random.permutation(len(self.captions))
        self.index_counter = 0
        self.epoch = 1
        self.raw_captions = raw_captions
        self.preload_features = preload_features
        if self.preload_features:
            self.img_features = np.array([np.load( '{0}.npz'.format(os.path.join(self.img_feature_root, os.path.splitext(image['file_path'])[0])))['arr_0'] for image in self.images ])
   
    def get_batch(self, batch_size=248, raw_img = False):
        print(self.index_counter, ' - ', self.index_counter + batch_size)
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
                batch_images = self.img_features[[self.cap2img[i] for i in batch_caption_indices ]]
            else:
                batch_images = np.array([ np.load( '{0}.npz'.format(os.path.join(self.img_feature_root, os.path.splitext(self.images[self.cap2img[i]]['file_path'])[0] )))['arr_0'] for i in batch_caption_indices ])
        
        if self.raw_captions:
            batch_word_indices = [ self.captions[i]['caption'] for i in batch_caption_indices ]
        else:
            batch_word_indices = [ np.array(self.captions[i]['caption'], dtype=np.int32) for i in batch_caption_indices]

        return batch_images, batch_word_indices

    def shuffle_data(self):
        self.random_indices = np.random.permutation(len(self.captions))

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
    with open('../data/captions/processed/dataset_STAIR_jp.pkl', 'br') as f:
        data = pickle.load(f)

    train_data = data['train']
    dataset = DataLoader(train_data, img_feature_root='../data/images/features/ResNet50/', img_root="../data/images/original/")
    batch_images, batch_word_indices = dataset.get_batch(10, raw_img=True)
    
    for img, words in zip(batch_images, batch_word_indices):
        print('img:', img)
        print('words', words)

    batch_images, batch_word_indices = dataset.get_batch(10)
    
    for ing, words in zip(batch_images, batch_word_indices):
        print('img:', img)
        print('words', words)

    dataset = DataLoader(train_data, img_feature_root='../data/images/features/ResNet50/', preload_features=True)
    batch_images, batch_word_indices = dataset.get_batch(10)
    for ing, words in zip(batch_images, batch_word_indices):
        print('img:', img)
        print('words', words)
