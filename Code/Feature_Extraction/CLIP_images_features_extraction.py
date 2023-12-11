import json
import os
import clip
from PIL import Image
import pickle
import torch

@torch.no_grad()
def feature_extraction(datasets, encoder, proprecess, annotations, outpath):
    # print('inside')
    results = []

    with open(annotations, 'r') as infile:
        annotations = json.load(infile)
    
    if os.path.exists(outpath):
        return
    
    if datasets == 'coco' or datasets == 'flickr30k': # coco, flickr30k
        # format = {image_path: [caption1, caption2, ...]} -> [[image_path, image_features, [caption1, caption2, ...]], ...]
        if datasets == 'coco':
            rootpath = '../../annotations/coco/val2014/'
        elif datasets == 'flickr30k':
            rootpath = '../../annotations/flickr30k/flickr30k-images/'

        for image_id in annotations:
            caption = annotations[image_id]
            image_path = rootpath + image_id
            image = proprecess(Image.open(image_path)).unsqueeze(dim = 0).to(device)
            image_features = encoder.encode_image(image).squeeze(dim = 0).to('cpu') # clip_hidden_size
            results.append([image_id, image_features, caption])

    else: # nocaps
        # format = [{'split': 'near_domain', 'image_id': '4499.jpg', 'caption': [caption1, caption2, ...]}, ...]
        # format = [[image_path, image_split, image_features, [caption1, captions2, ...]], ...]
        rootpath = '../../annotations/nocaps/'
        for annotation in annotations:
            split = annotation['split']
            image_id = annotation['image_id']
            caption = annotation['caption']
            image_path = rootpath + split + '/' + image_id
            image = proprecess(Image.open(image_path)).unsqueeze(dim = 0).to(device)
            image_features = encoder.encode_image(image).squeeze(dim = 0).to('cpu') # clip_hidden_size
            results.append([image_id, split, image_features, caption])

    with open(outpath, 'wb') as outfile:
        pickle.dump(results, outfile)

def util_feature_extraction():

    feature_extraction('nocaps', encoder, proprecess, path_nocaps, outpath_nocaps)

    feature_extraction('coco', encoder, proprecess, path_val_coco, outpath_val_coco)

    feature_extraction('coco', encoder, proprecess, path_test_coco, outpath_test_coco)

    feature_extraction('flickr30k', encoder, proprecess, path_val_flickr30k, outpath_val_flickr30k)

    feature_extraction('flickr30k', encoder, proprecess, path_test_flickr30k, outpath_test_flickr30k)


if __name__ == '__main__':

    device = 'cuda:0'
    clip_type = 'ViT-B/32'
    clip_name = clip_type.replace('/', '')

    path_nocaps = '../../annotations/nocaps/nocaps_corpus.json'
    path_val_coco = '../../annotations/coco/val_captions.json'
    path_test_coco = '../../annotations/coco/test_captions.json'
    path_val_flickr30k = '../../annotations/flickr30k/val_captions.json'
    path_test_flickr30k = '../../annotations/flickr30k/test_captions.json'

    outpath_nocaps = f'../../annotations/nocaps/nocaps_corpus_{clip_name}.pickle'
    outpath_val_coco = f'../../annotations/coco/val_captions_{clip_name}.pickle'
    outpath_test_coco = f'../../annotations/coco/test_captions_{clip_name}.pickle'
    outpath_val_flickr30k = f'../../annotations/flickr30k/val_captions_{clip_name}.pickle'
    outpath_test_flickr30k = f'../../annotations/flickr30k/test_captions_{clip_name}.pickle'

    encoder, proprecess = clip.load(clip_type, device)

    util_feature_extraction()