import os
import json
import argparse

class Search_Dataset(object):
    def __init__(self, json_dataset):
        with open(json_dataset) as f:
            self.dataset = json.load(f)
        
        self.licenses = self.dataset['licenses']
        self.annotations = self.dataset['annotations']
        self.images = self.dataset['images']
        self.info = self.dataset['info']
        self.img_num = len(self.dataset['images'])
        self.list_ids = []

    def search_img(self, img_id):

        data = 'Detail:\n'
        annot_data = []
        image_data = []

        for annotation in self.annotations:
            if annotation['image_id'] == int(img_id):
                annot_data.append(annotation)

        for image in self.images:
            if image['id'] == int(img_id):
                image_data = image
        
        if image_data:
            data += 'id: {0}\nflickr_url: {1}\ncoco_url: {2}\nfile_name : {3}\nheight: {4}\nwidth: {5}\n\n'.format(image_data['id'], image_data['flickr_url'], image_data['coco_url'], image_data['file_name'], image_data['height'], image_data['width'])
            
            data += 'The number of captions is {0}\n\n'.format(len(annot_data))
            for i, annot in enumerate(annot_data):
                data += 'caption{0}:\nid: {1}\nimage_id: {2}\ncaption: {3}\ntokenized: {4}\n\n'.format(i, annot['id'], annot['image_id'], annot['caption'], annot['tokenized_caption'])
                
            return data

        else:
            return 'Not Found'

    def search_captions(self, img_id):
        annot_data = []

        for annotation in self.annotations:
            if annotation['image_id'] == int(img_id):
                annot_data.append(annotation['tokenized_caption'])

        return annot_data

    def show_listid(self):
        if not self.list_ids:
            for img in self.images:
                self.list_ids.append(img['id'])
                
        return self.list_ids

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="search image and captions from id")
    parser.add_argument('--dict', '-d', type=str, default="../data/captions/stair_captions_v1.1_train.json")
    parser.add_argument('id', type=int, 
                        help="input id of image you want to search")
    args = parser.parse_args() 
    
    searcher = Search_Dataset(args.dict)
    result = searcher.search_img(args.id)
    print(result)
