# Address Element Extraction
## Shopee Code League 2021 - Data Science Problem

We are happy to share our solution, [ranked 28th at the private leaderboard](https://www.kaggle.com/c/scl-2021-ds/leaderboard), for Shopee Code League 2021 - Data Science Problem.

In this competition, the task is building a model to correctly extract Point of Interest (POI) Names and Street Names from unformatted Indonesia addresses collected by Shopee.

**Sample Dataset**

id | raw_address | POI/street
--- | --- | --- 
1 | karang mulia bengkel mandiri motor raya bosnik 21 blak kota | bengkel mandiri motor/raya bosnik
2 | primkob pabri adiwerna | primkob pabri/
3 | jalan mh thamrin, sei rengas i kel. medan kota | /jalan mh thamrin
4 | smk karya pemban, pon | smk karya pembangunan/pon

Examples - assume that:
1) The POI is "bengkel mandiri motor" and street name is "raya bosnik" the returned
POI/street should be:
    - bengkel mandiri motor/raya bosnik
2) The POI is "primkob pabri" and no street name is found the returned POI/street should
be:
    - primkob pabri/
3) No POI is found and the street name is "jalan mh thamrin" the returned POI/street should
be:
    - /jalan mh thamrin
4) The word "pembangunan" in raw_address "smk karya pemban, pon" is not complete. The
correct POI will be "smk karya pembangunan" and the returned result should be:
    - smk karya pembangunan/pon

## Our Solution

There are 3 main part of our solution :
* Text Preprocessing -> we remove multiple whitespace, restructure punctutaion, etc
* Text Repairing -> we use a probabilistic model, based on ngram, to repair some raw texts based on the train data
* Named Entity Recognition -> we fine-tune IndoBERT model for this task. We create our own labelling format, to match with Shopee DS problem with POI and street.

### 1. Installation

We run our experiment using Docker, started with [huggingface/transformers-pytorch-gpu:3.4.0](https://hub.docker.com/layers/huggingface/transformers-pytorch-gpu/3.4.0/images/sha256-7e0b2f97aad355f92b27063eef4245ac58e69e8c2113ea9bb0be6b4db23d301a?context=explore) image. You can pull the docker using this command 

```bash
docker pull huggingface/transformers-pytorch-gpu:3.4.0
```

After running the image as a container, please install some libraries needed

```bash
bash install.sh
```

### 2. Text Preprocessing
`Text-Preprocessing.ipynb`

### 3. Probabilistic Text Repair
`Probabilistic-Text-Repair.ipynb`

### 4. Fine-Tune IndoBERT for Shopee Task

**a. Create Custom Label for NER Model**

We implement a new data format at `indonlu/utils/data_utils.py` that match with Shopee problem in handling extraction of POI and street.

**b. Convert Train and Test data**

Create Train and Validation Data
```python
python3 create_train_label.py
```

Create Test Data
```python
python3 create_test_label.py
```

**c. Fine-Tuning**
```python
python3 train.py
```

**d. Prepare for Submission**
```python
python3 eval.py
```

### Performance

```bash
(Epoch 15) TRAIN LOSS:0.0019 ACC:1.00 F1:1.00 REC:1.00 PRE:1.00 LR:0.00000500
(Epoch 15) VALID LOSS:0.1422 ACC:0.98 F1:0.94 REC:0.94 PRE:0.94
save model checkpoint at models/bert-large/32_128_3e-05/
(Epoch 16) TRAIN LOSS:0.0020 ACC:1.00 F1:1.00 REC:1.00 PRE:1.00 LR:0.00000500
(Epoch 16) VALID LOSS:0.1394 ACC:0.98 F1:0.94 REC:0.94 PRE:0.94
save model checkpoint at models/bert-large/32_128_3e-05/
(Epoch 17) TRAIN LOSS:0.0019 ACC:1.00 F1:1.00 REC:1.00 PRE:1.00 LR:0.00000500
(Epoch 17) VALID LOSS:0.1440 ACC:0.98 F1:0.94 REC:0.94 PRE:0.94
save model checkpoint at models/bert-large/32_128_3e-05/
```

Our fine-tuned model performs very well, even we got an average score at Kaggle. We think that the *strict* metric for measuring the performance is the key reason why our submission is not good enough.

Thanks for reading, please don't hestitate to contact me, mhilmiasyrofi(at)gmail(dot)com, if you need further assistance in repplicating the result!


