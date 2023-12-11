import json

# format
# {
#     'filepath': 'val2014',
#     'sentids': [770337, 771687, 772707, 776154, 781998],
#     'filename': 'COCO_val2014_000000391895.jpg',
#     'imgid': 0,
#     'split': 'test',
#     'sentences': [
#         {'tokens': ['a', 'man', 'with', 'a', 'red', 'helmet', 'on', 'a', 'small', 'moped', 'on', 'a', 'dirt', 'road'],
#         'raw': 'A man with a red helmet on a small moped on a dirt road. ',
#         'imgid': 0,
#         'sentid': 770337},
#         {'tokens': ['man', 'riding', 'a', 'motor', 'bike', 'on', 'a', 'dirt', 'road', 'on', 'the', 'countryside'],
#         'raw': 'Man riding a motor bike on a dirt road on the countryside.',
#         'imgid': 0,
#         'sentid': 771687},
#         {'tokens': ['a', 'man', 'riding', 'on', 'the', 'back', 'of', 'a', 'motorcycle'],
#         'raw': 'A man riding on the back of a motorcycle.',
#         'imgid': 0,
#         'sentid': 772707},
#         {'tokens': ['a', 'dirt', 'path', 'with', 'a', 'young', 'person', 'on', 'a', 'motor', 'bike', 'rests', 'to', 'the', 'foreground', 'of', 'a', 'verdant', 'area', 'with', 'a', 'bridge', 'and', 'a', 'background', 'of', 'cloud', 'wreathed', 'mountains'],
#         'raw': 'A dirt path with a young person on a motor bike rests to the foreground of a verdant area with a bridge and a background of cloud-wreathed mountains. ',
#         'imgid': 0,
#         'sentid': 776154},
#         {'tokens': ['a', 'man', 'in', 'a', 'red', 'shirt', 'and', 'a', 'red', 'hat', 'is', 'on', 'a', 'motorcycle', 'on', 'a', 'hill', 'side'],
#         'raw': 'A man in a red shirt and a red hat is on a motorcycle on a hill side.',
#         'imgid': 0,
#         'sentid': 781998}],
#     'cocoid': 391895
# }

# format
# {

#     'filename': 'COCO_val2014_000000391895.jpg',
#     'split': 'test',
# }

def main():
    
    with open('./karpathy_split/dataset.json') as infile:
        annotations = json.load(infile) # {images: List, dataset: 'coco'}
    images = annotations['images']
    
    # splits -> 'train', 'restval', 'val', 'test'
    train_split = {}
    val_split = {}
    test_split = {}
    for image in images:
        captions = []
        for caption in image['sentences']:
            captions.append(caption['raw'])
        if image['split'] == 'train' or image['split'] == 'restval':
            train_split[image['filename']] = captions
        elif image['split'] == 'val':
            val_split[image['filename']] = captions
        elif image['split'] == 'test':
            test_split[image['filename']] = captions

    with open('./train_captions.json', 'w') as outfile:
        json.dump(train_split, outfile, indent = 4)
    with open('./val_captions.json', 'w') as outfile:
        json.dump(val_split, outfile, indent = 4)
    with open('./test_captions.json', 'w') as outfile:
        json.dump(test_split, outfile, indent = 4)

if __name__ == '__main__':
    main()