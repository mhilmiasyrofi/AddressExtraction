# Address Element Extraction
## Shopee Code League 2021 - Data Science Problem

We are happy to share our solution, [ranked 28th at the private leaderboard](https://www.kaggle.com/c/scl-2021-ds/leaderboard), for Shopee Code League 2021 - Data Science Problem

There are 3 main part of our solution :
* Text Preprocessing
* Text Repairing -> we use a probabilistic model, based on ngram, to repair some raw texts based on the train data
* Named Entity Recognition -> we fine-tune IndoBERT model for this task. We create our own labelling format, to match with Shopee DS problem.

### Installation

We run our experiment using Docker, started with [huggingface/transformers-pytorch-gpu:3.4.0](https://hub.docker.com/layers/huggingface/transformers-pytorch-gpu/3.4.0/images/sha256-7e0b2f97aad355f92b27063eef4245ac58e69e8c2113ea9bb0be6b4db23d301a?context=explore) image. You can pull the docker using this command 

```
docker pull huggingface/transformers-pytorch-gpu:3.4.0
```

After running the image as a container, please install some libraries needed

```
bash install.sh
```

### Text Preprocessing
`Text-Preprocessing.ipynb`

### Probabilistic Text Repair
`Probabilistic-Text-Repair.ipynb`

### Fine-Tune IndoBERT for Downstream Task

**1. Create Custom Label for NER Model**

We implement a new data format at `indonlu/utils/data_utils.py` that match with Shopee problem in handling POI and street.

**2. Convert Train and Test data**

Create Train and Validation Data
```
python3 create_train_label.py
```

Create Test Data
```
python3 create_test_label.py
```

**Fine-Tuning**
```
python3 run.py
```

**Prepare for Submission**
```
python3 eval.py
```
**Performance**
```
[2021/03/20 05:07:57] - TRAIN LOSS:0.0535 ACC:0.99 F1:0.95 REC:0.96 PRE:0.95 LR:0.00000500
[2021/03/20 05:08:08] - save model checkpoint at models/bert-large/32_128_2e-05/
[2021/03/20 05:12:58] - VALID LOSS:0.0429 ACC:0.99 F1:0.96 REC:0.97 PRE:0.96
```
Our fine-tuned model performs very well, even we got an average score at Kaggle. We think that the *strict* metric for measuring the performance is the key reason why our submission is not good enough.

Thanks for reading, please don't hestitate to contact me, mhilmiasyrofi(at)gmail(dot)com, if you need further assistance in repplicating the result!


