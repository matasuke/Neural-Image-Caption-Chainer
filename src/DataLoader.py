import numpy as np
import os
from img_proc import Img_proc

#get_batch_val doesn't work well when getting batch second times,
#the size of batch is (0, 2048) not (256, 2048)


class DataLoader:
    def __init__(self, dataset, img_feature_root=os.path.join('..', 'data', 'images', 'features', 'ResNet50',), img_root=os.path.join('..', 'data', 'images', 'original'), img_mean="imagenet", preload_features=False, exist_test = True):
        self.train = dataset['train']
        self.train_captions = self.train['captions']
        self.train_images = self.train['images']
        self.num_train_captions = len(self.train_captions)
        self.num_train_images = len(self.train_images)
        self.train_index_counter = 0
        self.train_cap2img = { caption['caption_idx']:caption['img_idx'] for caption in self.train_captions }
       
        if exist_test:
            self.val = dataset['val']
            self.val_captions = self.val['captions']
            self.val_images = self.val['images']
            self.num_val_captions = len(self.val_captions)
            self.num_val_images = len(self.val_images)
            self.val_index_counter = 0
            self.val_cap2img = { caption['caption_idx']:caption['img_idx'] for caption in self.val_captions }

            self.test = dataset['test']
            self.test_captions = self.test['captions']
            self.test_images = self.test['images']
            self.num_test_captions = len(self.test_captions)
            self.num_test_images = len(self.test_images)
            self.test_index_counter = 0
            self.test_cap2img = { caption['caption_idx']:caption['img_idx'] for caption in self.test_captions }
        
        self.token2index = dataset['word_index']
        self.index2token = { v: k for k, v in self.token2index.items() }
        self.num_tokens = len(self.token2index)
        
        self.img_proc = Img_proc(mean_type=img_mean)
        self.img_feature_root = img_feature_root
        self.img_root = img_root
        self.random_indices = np.random.permutation(self.num_train_captions)
        self.epoch = 1
        self.continue_val = True
        self.continue_test = True
        self.preload_features = preload_features
        
        if self.preload_features:
            self.img_features_train = np.array([np.load( '{0}.npz'.format(os.path.join(self.img_feature_root, os.path.splitext(image['file_path'])[0])))['arr_0'] for image in self.train_images ])
            if exist_test:
                self.img_features_val = np.array([np.load( '{0}.npz'.format(os.path.join(self.img_feature_root, os.path.splitext(image['file_path'])[0])))['arr_0'] for image in self.val_images ])
                self.img_features_test = np.array([np.load( '{0}.npz'.format(os.path.join(self.img_feature_root, os.path.splitext(image['file_path'])[0])))['arr_0'] for image in self.test_images ])

    def get_batch_train(self, batch_size=256, raw_img = False, raw_captions=False):
        batch_caption_indices = self.random_indices[self.train_index_counter: self.train_index_counter + batch_size]
        self.train_index_counter += batch_size
        
        if self.train_index_counter > self.num_train_captions - batch_size:
            self.epoch += 1
            self.shuffle_data()
            self.train_index_counter = 0

        if raw_img:
            batch_images = np.array([ self.img_proc.load_img(os.path.join(self.img_root, self.train_images[self.train_cap2img[i]]['file_path']), expand_dim = False) for i in batch_caption_indices ])
        else:
            if self.preload_features:
                batch_images = self.img_features_train[[self.train_cap2img[i] for i in batch_caption_indices ]]
            else:
                batch_images = np.array([ np.load( '{0}.npz'.format(os.path.join(self.img_feature_root, os.path.splitext(self.train_images[self.train_cap2img[i]]['file_path'])[0] )))['arr_0'] for i in batch_caption_indices ])
        
        if raw_captions:
            batch_word_indices = [ self.train_captions[i]['caption'] for i in batch_caption_indices ]
        else:
            batch_word_indices = [ np.array(self.train_captions[i]['caption'], dtype=np.int32) for i in batch_caption_indices]

        return batch_images, batch_word_indices

    def get_batch_raw(self, batch_size=256, raw_img=False):
        batch_caption_indices = self.random_indices[self.train_index_counter: self.train_index_counter + batch_size]
        self.train_index_counter += batch_size
        
        if self.train_index_counter > self.num_train_captions - batch_size:
            self.epoch += 1
            self.shuffle_data()
            self.train_index_counter = 0

        batch_images_path = [ self.train_images[self.train_cap2img[i]]['file_path' ] for i in batch_caption_indices ]

        batch_word_indices = [ self.train_captions[i]['caption'] for i in batch_caption_indices ]

        for i, cap in enumerate(batch_word_indices):
            for j, c in enumerate(cap):
                batch_word_indices[i][j] = self.index2token[c]

        return batch_images_path, batch_word_indices


    def get_batch_val(self, batch_size=256, raw_img = False, raw_captions = False):
        batch_caption_indices = list(range(self.val_index_counter, batch_size + self.val_index_counter))
        self.val_index_counter += batch_size
    
        if self.val_index_counter > self.num_val_captions -  batch_size:
            self.val_index_counter = 0
            self.continue_val = False
        
        if raw_img:
            batch_images = np.array([ self.img_proc.load_img(os.path.join(self.img_root, self.val_images[self.val_cap2img[i]]['file_path']), expand_dim = False) for i in batch_caption_indices ])
        else:
            if self.preload_features:
                batch_images = self.img_features_val[[self.val_cap2img[i] for i in batch_caption_indices ]]
            else:
                batch_images = np.array([ np.load( '{0}.npz'.format(os.path.join(self.img_feature_root, os.path.splitext(self.val_images[self.val_cap2img[i]]['file_path'])[0] )))['arr_0'] for i in batch_caption_indices ])

        if raw_captions:
            batch_word_indices = [ self.val_captions[i]['caption'] for i in batch_caption_indices ]
        else:
            batch_word_indices = [ np.array(self.val_captions[i]['caption'], dtype=np.int32) for i in batch_caption_indices]

        return batch_images, batch_word_indices

    def get_batch_test(self, batch_size=256, raw_img = False, raw_captions = False):
        batch_caption_indices = list(range(self.test_index_counter, batch_size + self.test_index_counter))
        self.test_index_counter += batch_size
    
        if self.test_index_counter > self.num_test_captions -  batch_size:
            self.test_index_counter = 0
            self.continue_test = False
        
        if raw_img:
            batch_images = np.array([ self.img_proc.load_img(os.path.join(self.img_root, self.test_images[self.test_cap2img[i]]['file_path']), expand_dim = False) for i in batch_caption_indices ])
        else:
            if self.preload_features:
                batch_images = self.img_features_test[[self.test_cap2img[i] for i in batch_caption_indices ]]
            else:
                batch_images = np.array([ np.load( '{0}.npz'.format(os.path.join(self.img_feature_root, os.path.splitext(self.test_images[self.test_cap2img[i]]['file_path'])[0] )))['arr_0'] for i in batch_caption_indices ])

        if raw_captions:
            batch_word_indices = [ self.test_captions[i]['caption'] for i in batch_caption_indices ]
        else:
            batch_word_indices = [ np.array(self.test_captions[i]['caption'], dtype=np.int32) for i in batch_caption_indices]

        return batch_images, batch_word_indices
    
    def shuffle_data(self):
        self.random_indices = np.random.permutation(self.num_train_captions)
    
    @property
    def dict_size(self):
        return self.num_tokens

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
    
    batch_images, batch_word_indices = dataset.get_batch_train(10, raw_img=False, raw_captions=False)
    for img, words in zip(batch_images, batch_word_indices):
        print('img:', img)
        print('words', words)
    
    batch_images, batch_word_indices = dataset.get_batch_val(10, raw_img=False, raw_captions=False)
    for img, words in zip(batch_images, batch_word_indices):
        print('img:', img)
        print('words', words)

    batch_images, batch_word_indices = dataset.get_batch_test(10, raw_img=False, raw_captions=False)
    for img, words in zip(batch_images, batch_word_indices):
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

