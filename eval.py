from utils.data_utils import NerGritDataset, NerDataLoader
from utils.metrics import ner_metrics_fn
from utils.forward_fn import forward_word_classification
from modules.word_classification import BertForWordClassification
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
    model_epoch = 1

    model_name = "indobenchmark/indobert-{}-p1".format(model_version)

    # Load Tokenizer and Config
    tokenizer = BertTokenizer.from_pretrained(model_name)
    config = BertConfig.from_pretrained(model_name)
    config.num_labels = NerGritDataset.NUM_LABELS

    train_dataset_path = 'data/bert-fine-tune/train.txt'
    valid_dataset_path = 'data/bert-fine-tune/validation.txt'

    train_dataset = NerGritDataset(
        train_dataset_path, tokenizer, lowercase=True)
    valid_dataset = NerGritDataset(
        valid_dataset_path, tokenizer, lowercase=True)

    batch_size = 32
    max_seq_len = 128

    train_loader = NerDataLoader(dataset=train_dataset, max_seq_len=max_seq_len,
                                 batch_size=batch_size, num_workers=16, shuffle=True)
    valid_loader = NerDataLoader(dataset=valid_dataset, max_seq_len=max_seq_len,
                                 batch_size=batch_size, num_workers=16, shuffle=False)

    w2i, i2w = NerGritDataset.LABEL2INDEX, NerGritDataset.INDEX2LABEL

    # Instantiate model
    model = BertForWordClassification.from_pretrained(
        model_name, config=config)

    output_model = "models/bert-{}/model-{}.pth".format(
        model_version, model_epoch)
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
            if labels[i] == "B-PLACE":
                street += text[i] + " "
                i += 1
                while i < len(labels) and (labels[i] == "B-PLACE" or labels[i] == "I-PLACE"):
                    street += text[i] + " "
                    i += 1
                street = street[:-1]
            elif labels[i] == "B-ORGANISATION":
                poi += text[i] + " "
                i += 1
                while i < len(labels) and (labels[i] == "B-ORGANISATION" or labels[i] == "I-ORGANISATION"):
                    poi += text[i] + " "
                    i += 1
                poi = poi[:-1]
            else:
                i += 1
        return "{}/{}".format(poi, street)

    df = pd.read_csv("data/test.csv")
    df["POI/street"] = df["raw_address"].apply(extract_poi_street)
    df = df.drop(columns=["raw_address"])

    SGT = pytz.timezone('Singapore')
    datetime_sgt = datetime.now(SGT)
    time_now = datetime_sgt.strftime('%Y:%m:%d-%H:%M:%S')

    csv_path = "submissions/bert-{}/{}.csv".format(model_version, time_now)
    print("File saved at: ", csv_path)
    df.to_csv(csv_path, index=False)
