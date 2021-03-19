import os
import sys

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

from utils.data_utils import NerShopeeDataset, NerDataLoader
from utils.metrics import ner_metrics_fn
from utils.forward_fn import forward_word_classification
from modules.word_classification import BertForWordClassification

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


if __name__ == "__main__":

    # Set random seed
    set_seed(26092020)

    model_version = "large"
    # model_version = "base"
    model_epoch = 1

    model_name = "indobenchmark/indobert-{}-p1".format(model_version)

    # Load Tokenizer and Config
    tokenizer = BertTokenizer.from_pretrained(model_name)
    config = BertConfig.from_pretrained(model_name)
    config.num_labels = NerShopeeDataset.NUM_LABELS

    w2i, i2w = NerShopeeDataset.LABEL2INDEX, NerShopeeDataset.INDEX2LABEL

    # Instantiate model
    model = BertForWordClassification.from_pretrained(
        model_name, config=config)

    # model_version == "base" :
    batch_size = 16
    eval_batch_size = 16
    max_seq_len = 128
    if model_version == "large":
        batch_size = 32
        eval_batch_size = 32
        max_seq_len = 128
    
    learning_rate = 1e-6
    if model_version == "large" :
        learning_rate = 2e-5 

    model_dir = "models/bert-{}/".format(model_version)
    model_dir = "{}{}_{}_{}/".format(model_dir, batch_size, max_seq_len, learning_rate)

    output_model = "{}model-{}.pth".format(
        model_dir, model_epoch)
    print("Loaded model: ", output_model)
    checkpoint = torch.load(output_model, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    model = model.cuda()

    # Evaluate on validation
    model.eval()
    torch.set_grad_enabled(False)

    def extract_poi_street(text):
        if text == "":
            return "/"
        text = word_tokenize(text)
        subwords, subword_to_word_indices = word_subword_tokenize(
            text, tokenizer)

        subwords = torch.LongTensor(subwords).view(1, -1).to(model.device)
        subword_to_word_indices = torch.LongTensor(
            subword_to_word_indices).view(1, -1).to(model.device)
        logits = model(subwords, subword_to_word_indices)[0]

        preds = torch.topk(logits, k=1, dim=-1)[1].squeeze().cpu().numpy()
        if preds.size == 1:
            preds = np.array([preds])
        labels = [i2w[preds[i]] for i in range(len(preds))]

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
        return "{}/{}".format(poi, street)


    df = pd.read_csv("data/cleaned_test.csv")
    df["POI/street"] = df["raw_address"].apply(extract_poi_street)
    df = df.drop(columns=["raw_address"])

    SGT = pytz.timezone('Singapore')
    datetime_sgt = datetime.now(SGT)
    time_now = datetime_sgt.strftime('%Y-%m-%d--%H-%M-%S')

    csv_path = "submissions/bert-{}/{}.csv".format(model_version, time_now)
    print("File saved at: ", csv_path)
    df.to_csv(csv_path, index=False)
