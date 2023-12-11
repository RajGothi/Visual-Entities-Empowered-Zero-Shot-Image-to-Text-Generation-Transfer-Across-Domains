import json

# format
# {
#     'sentids': [0, 1, 2, 3, 4],
#     'imgid': 0,
#     'sentences': [
#         {'tokens': ['two', 'young', 'guys', 'with', 'shaggy', 'hair', 'look', 'at', 'their', 'hands', 'while', 'hanging', 'out', 'in', 'the', 'yard'],
#          'raw': 'Two young guys with shaggy hair look at their hands while hanging out in the yard.',
#          'imgid': 0,
#          'sentid': 0},
#         {'tokens': ['two', 'young', 'white', 'males', 'are', 'outside', 'near', 'many', 'bushes'],
#          'raw': 'Two young, White males are outside near many bushes.',
#          'imgid': 0,
#          'sentid': 1},
#         {'tokens': ['two', 'men', 'in', 'green', 'shirts', 'are', 'standing', 'in', 'a', 'yard'],
#          'raw': 'Two men in green shirts are standing in a yard.', 
#          'imgid': 0,
#          'sentid': 2},
#         {'tokens': ['a', 'man', 'in', 'a', 'blue', 'shirt', 'standing', 'in', 'a', 'garden'],
#          'raw': 'A man in a blue shirt standing in a garden.',
#          'imgid': 0,
#          'sentid': 3},
#         {'tokens': ['two', 'friends', 'enjoy', 'time', 'spent', 'together'],
#          'raw': 'Two friends enjoy time spent together.',
#          'imgid': 0,
#          'sentid': 4}
#     ],
#     'split': 'train',
#     'filename': '1000092795.jpg'
# }

# format
# {

#     'filename': 'COCO_val2014_000000391895.jpg',
#     'split': 'test',
# }

def main():
    
    # with open('./karpathy_split/dataset.json') as infile:
    with open('./karpathy_split/dataset.json') as infile:
        annotations = json.load(infile) # {images: List, dataset: 'flickr30k'}
    images = annotations['images']
    

    # splits -> 'train', 'val', 'test'
    train_split = {}
    val_split = {}
    test_split = {}
    for image in images:
        captions = []
        for caption in image['sentences']:
            captions.append(caption['raw'])
        if image['split'] == 'train':
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