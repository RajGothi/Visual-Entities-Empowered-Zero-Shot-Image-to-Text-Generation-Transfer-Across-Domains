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
bash eval_flickr30k.sh coco_train_1 0 '--top_k 3 --threshold 0.2' 14
bash eval_coco.sh flicker30K_1 0 '--top_k 3 --threshold 0.2 --using_greedy_search' 29
```

- In-Domain
```bash
bash eval_coco.sh coco_train_1 0 '' 14
bash eval_flickr30k.sh flicker30K_1 0 '' 29
```

# Other:
- PPT Link: https://docs.google.com/presentation/d/1MY__ajJE0VolzEsR9XVgSn5Dtjhx1VWo4irGBco-WsY/edit#slide=id.g28910898772_1_38
- If checkpoints are required to run the inference, I can share them privately.(Email me on rajgothi9@gmail.com)
- The 'logs' folder contains the experiment logs generated after running.
- The 'annotations' folder holds all the downloaded dataset information, which consists of large files. If necessary, I can share this information privately.
- The 'checkpoints' folder includes checkpoints from each individual experiment, which have generated captions.
