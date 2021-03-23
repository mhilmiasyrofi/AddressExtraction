import os
import sys
import logging

import time
from datetime import datetime
import pytz

import random
import numpy as np
import pandas as pd
import torch
from torch import optim
from tqdm import tqdm

from transformers import BertConfig, BertTokenizer
from transformers import get_linear_schedule_with_warmup

from nltk.tokenize import word_tokenize

sys.path.append("indonlu")

from modules.word_classification import BertForWordClassification
from utils.forward_fn import forward_word_classification
from utils.metrics import ner_metrics_fn
from utils.data_utils import NerShopeeDataset, NerDataLoader


## custom time zone for logger

def customTime(*args):
    utc_dt = pytz.utc.localize(datetime.utcnow())
    converted = utc_dt.astimezone(pytz.timezone("Singapore"))
    return converted.timetuple()


###
# common functions
###

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def count_param(module, trainable=False):
    if trainable:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in module.parameters())


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def metrics_to_string(metric_dict):
    string_list = []
    for key, value in metric_dict.items():
        string_list.append('{}:{:.2f}'.format(key, value))
    return ' '.join(string_list)


def word_subword_tokenize(sentence, tokenizer):
    # Add CLS token
    subwords = [tokenizer.cls_token_id]
    subword_to_word_indices = [-1]  # For CLS

    # Add subwords
    for word_idx, word in enumerate(sentence):
        subword_list = tokenizer.encode(word, add_special_tokens=False)
        subword_to_word_indices += [word_idx for i in range(len(subword_list))]
        subwords += subword_list

    # Add last SEP token
    subwords += [tokenizer.sep_token_id]
    subword_to_word_indices += [-1]

    return subwords, subword_to_word_indices


def save(fpath, model, optimizer):
    # save
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, fpath)

## drop the last label because we add dot "." in the last
def drop_last(raw, arr):
    if raw.split(" ")[-1] == ".":
        return arr
    return arr[:-1]


def post_process(text):
    return text.replace(" , ", ", ").replace(" . ", ". ")


def extract_poi_street(text, labels):
    if text == "":
        return "/"

    text = text.split(" ")

    poi = ""
    street = ""
    i = 0
    while i < len(text):
        if labels[i] == "B-POI":
            poi += text[i] + " "
            i += 1
            while i < len(labels) and (labels[i] == "B-POI" or labels[i] == "I-POI"):
                poi += text[i] + " "
                i += 1
            poi = poi[:-1]
        elif labels[i] == "B-STREET":
            street += text[i] + " "
            i += 1
            while i < len(labels) and (labels[i] == "B-STREET" or labels[i] == "I-STREET"):
                street += text[i] + " "
                i += 1
            street = street[:-1]
        else:
            i += 1
    poi = post_process(poi)
    street = post_process(street)
    return "{}/{}".format(poi, street)



if __name__ == "__main__" :
    # Set random seed
    set_seed(26092020)

    # model_version = "base"
    model_version = "large"
    use_regularization = False
    
    model_epoch = 17

    model_name = "indobenchmark/indobert-{}-p1".format(model_version)

    # model_version == "base" :
    batch_size = 32
    eval_batch_size = 16
    max_seq_len = 128
    if model_version == "large":
        batch_size = 32
        eval_batch_size = 32
        max_seq_len = 128

    learning_rate = 2e-5
    if model_version == "large":
        learning_rate = 3e-5

    model_dir = "models/bert-{}/".format(model_version)
    
    model_dir = "{}{}_{}_{}".format(
        model_dir, batch_size, max_seq_len, learning_rate)
    if use_regularization:
        model_dir += "_regularization"

    model_dir += "/"

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(model_dir, 'eval.log')),
            logging.StreamHandler()
        ])
    logging.Formatter.converter = customTime

    logger.info("Model: indobert-{}".format(model_version))
    logger.info("Batch Size: {}".format(batch_size))
    logger.info("Max Seq Length: {}".format(max_seq_len))
    logger.info("Learning Rate: {}".format(learning_rate))
    
    # Load Tokenizer and Config
    tokenizer = BertTokenizer.from_pretrained(model_name)
    config = BertConfig.from_pretrained(model_name)
    config.num_labels = NerShopeeDataset.NUM_LABELS

    w2i, i2w = NerShopeeDataset.LABEL2INDEX, NerShopeeDataset.INDEX2LABEL

    # Instantiate model
    model = BertForWordClassification.from_pretrained(
        model_name, config=config)

    
    output_model = "{}model-{}.pth".format(
        model_dir, model_epoch)
    logger.info("Loaded model: {}".format(output_model))
    checkpoint = torch.load(output_model, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    model = model.cuda()

    # Evaluate on validation
    model.eval()
    torch.set_grad_enabled(False)

    valid_dataset_path = 'data/bert-fine-tune/validation.txt'
    valid_dataset = NerShopeeDataset(valid_dataset_path, tokenizer, lowercase=True)

    valid_loader = NerDataLoader(dataset=valid_dataset, max_seq_len=max_seq_len,
                                batch_size=256, num_workers=16, shuffle=False)

    total_loss, total_correct, total_labels = 0, 0, 0
    list_hyp, list_label = [], []

    pbar = tqdm(valid_loader, leave=True, total=len(valid_loader))
    for i, batch_data in enumerate(pbar):
        batch_seq = batch_data[-1]
        loss, batch_hyp, batch_label = forward_word_classification(
            model, batch_data[:-1], i2w=i2w, device='cuda')

        # Calculate total loss
        valid_loss = loss.item()
        total_loss = total_loss + valid_loss

        # Calculate evaluation metrics
        list_hyp += batch_hyp
        list_label += batch_label
        metrics = ner_metrics_fn(list_hyp, list_label)

        pbar.set_description("VALID LOSS:{:.4f} {}".format(
            total_loss/(i+1), metrics_to_string(metrics)))

    metrics = ner_metrics_fn(list_hyp, list_label)

    logger.info("VALID LOSS:{:.4f} {}".format(
        total_loss/(i+1), metrics_to_string(metrics)))

    
    test_dataset_path = 'data/bert-fine-tune/test.txt'
    logger.info(test_dataset_path)
    test_dataset = NerShopeeDataset(test_dataset_path, tokenizer, lowercase=True)

    test_loader = NerDataLoader(dataset=test_dataset, max_seq_len=max_seq_len,
                                batch_size=256, num_workers=16, shuffle=False)

    total_loss, total_correct, total_labels = 0, 0, 0
    list_hyp, list_label = [], []

    pbar = tqdm(test_loader, leave=True, total=len(test_loader))
    for i, batch_data in enumerate(pbar):
        _, batch_hyp, _ = forward_word_classification(
            model, batch_data[:-1], i2w=i2w, device='cuda')
        list_hyp += batch_hyp
    
    df = pd.read_csv("data/cleaned_test.csv")
    df["label"] = list_hyp
    df["label"] = df.apply(lambda x: drop_last(x.raw_address, x.label), axis=1)
    df["POI/street"] = df.apply(
        lambda x: extract_poi_street(x.raw_address, x.label), axis=1)

    SGT = pytz.timezone('Singapore')
    datetime_sgt = datetime.now(SGT)
    time_now = datetime_sgt.strftime('%Y-%m-%d--%H-%M-%S')

    csv_path = "submissions/bert-{}/{}.csv".format(model_version, time_now)
    logger.info("File saved at: {}".format(csv_path))
    df[["id", "POI/street"]].to_csv(csv_path, index=False)

#     logger.info("Sanity Check with The Best Previous Submission")
#     path = "submissions/bert-large/2021-03-20--22-12-12.csv"
#     dfc = pd.read_csv(path)
#     check = dfc["POI/street"] == df["POI/street"]
#     logger.info("Similarity: {:.2f}%".format(100 * sum(check)/len(check)))
