# Visual Entities Empowered Zero-Shot Image-to-Text Generation Transfer Across Domains

### Problem statement:
- Given an image I, the goal is to generate a textual description using a pre-trained Vision-Language Model (VLM) while leveraging real-world knowledge from a Large Language Model (LLM). 
- The primary focus is on addressing challenges related to *modality bias* and *object hallucination*.

![Problem](Other/ps-ss.png)

### Method:
![method](Other/method-ss.png)
- Training: With text-only corpus, nouns are extracted from the sentence by a grammar parser to construct the hard prompt. Then, the soft prompt encodes the overall contexts of the sentence by CLIP text encoder.
- Inference: CLIP Image encoding pass to projector which gives soft prompt, along with that CLIP-based entity classifier to construct the entity-aware hard prompt. With the strong transferability from the training-agnostic hard prompt.



### Analysis:
![method](Other/analysis.png)

# Code:

### Requirements:
```bash
pip install .
git clone https://github.com/tylin/coco-caption
```

### Data Preparation:
```bash
cd Code/utils/
python get_entities.py
```

```bash
cd Code/Feature_Extraction/
python CLIP_texts_features_extraction.py
python CLIP_images_features_extraction.py
```

```bash
cd Code/utils/
python prompt_ensemble.py
```

### Training:

```bash
cd scripts/
bash train_coco.sh 0
bash train_flickr30k.sh 0
```
where 0 represent the GPU number.

### Inference:
To run the model on single image, you can use [Notebook](Code/Notebook/inference.ipynb)

### Evaluation:

- Cross-Domain
```bash
bash eval_nocaps.sh coco_train_1 0 '--top_k 3 --threshold 0.2' 14
```




PPT Link: https://docs.google.com/presentation/d/1MY__ajJE0VolzEsR9XVgSn5Dtjhx1VWo4irGBco-WsY/edit#slide=id.g28910898772_1_38

If Checkpoints are needed to run the inference, We can share it privately.

Scripts folder help to train the model and Validation on dataset using bash script.

Code/Notebook folder contain the inference file that can be infer on one instance.

Code/Utils have general functionality function that frequently used by other codes.

Code/Feature Extraction folder contain the code that help to extract the text and image features from the CLIP model.

Code/Model contains the model architecture that we used for our approach.

Train and validation py files are present in the Code folder.

Logs folder contain the logs of experiments after running it.

annotations folder contain the downlaoded all dataset information, It's too large files. (We could not able to share in Git/Moodle.)  (If needed then we can share privately.)

checkpoints folder contain the checkpoints of each indiviual experiments that we have run generated captions of those experiments.

Evaluation folder is cloned from the standard MS COCO git repo, which widely used in Captioning task for the evaluation. 
Please clone this git repo. (https://github.com/tylin/coco-caption)

Latest Readme can be find at (https://github.com/RajGothi/GNR-650/tree/main/Project)
