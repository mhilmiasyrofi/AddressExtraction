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
from nltk.tokenize import word_tokenize

sys.path.append("indonlu")

from utils.data_utils import NerGritDataset, NerDataLoader
from utils.metrics import ner_metrics_fn
from utils.forward_fn import forward_word_classification
from modules.word_classification import BertForWordClassification

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


def extract_poi_street(text):
    if text == "":
        return "/"
    text = word_tokenize(text)
    subwords, subword_to_word_indices = word_subword_tokenize(text, tokenizer)

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


def save(fpath, model, optimizer):
    # save
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, fpath)


if __name__ == "__main__":

    # Set random seed
    set_seed(26092020)

    # model_version = "base"
    model_version = "large"

    model_dir = "models/bert-{}/".format(model_version)

    if not os.path.exists(model_dir) :
        os.makedirs(model_dir)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(model_dir, 'output.log')),
            logging.StreamHandler()
        ])
    logging.Formatter.converter = customTime

    logger.info("Model: indobert-{}".format(model_version))

    model_name = "indobenchmark/indobert-{}-p1".format(model_version)

    # Load Tokenizer and Config
    tokenizer = BertTokenizer.from_pretrained(model_name)
    config = BertConfig.from_pretrained(model_name)
    config.num_labels = NerGritDataset.NUM_LABELS

    # Instantiate model
    model = BertForWordClassification.from_pretrained(
        model_name, config=config)

    train_dataset_path = 'data/bert-fine-tune/train.txt'
    valid_dataset_path = 'data/bert-fine-tune/validation.txt'

    train_dataset = NerGritDataset(
        train_dataset_path, tokenizer, lowercase=True)
    valid_dataset = NerGritDataset(
        valid_dataset_path, tokenizer, lowercase=True)

    # model_version == "base" :
    batch_size = 16
    eval_batch_size = 64

    if model_version == "large":
        batch_size = 32
        eval_batch_size = 64

    max_seq_len = 128

    train_loader = NerDataLoader(dataset=train_dataset, max_seq_len=max_seq_len,
                                 batch_size=batch_size, num_workers=16, shuffle=True)
    valid_loader = NerDataLoader(dataset=valid_dataset, max_seq_len=max_seq_len,
                                 batch_size=eval_batch_size, num_workers=16, shuffle=False)

    w2i, i2w = NerGritDataset.LABEL2INDEX, NerGritDataset.INDEX2LABEL

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]

    learning_rate = 2e-5
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model = model.cuda()

    # Train

    n_epochs = 3
    if model_version == "base":
        n_epochs = 4

    logger.info("Model: indobert-{}".format(model_version))
    logger.info("Batch Size: {}".format(batch_size))
    logger.info("Max Seq Length: {}".format(max_seq_len))
    logger.info("Learning Rate: {}".format(learning_rate))
    logger.info("Epochs: {}".format(n_epochs))
#     logger.info("{}".format())

    min_loss = sys.maxsize
    max_f1 = 0
    for epoch in range(n_epochs):
        model.train()
        torch.set_grad_enabled(True)

        total_train_loss = 0
        list_hyp, list_label = [], []

        train_pbar = tqdm(train_loader, leave=True, total=len(train_loader))
        for i, batch_data in enumerate(train_pbar):
            # Forward model
            loss, batch_hyp, batch_label = forward_word_classification(
                model, batch_data[:-1], i2w=i2w, device='cuda')

            # Update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tr_loss = loss.item()
            total_train_loss = total_train_loss + tr_loss

            # Calculate metrics
            list_hyp += batch_hyp
            list_label += batch_label

            train_pbar.set_description("(Epoch {}) TRAIN LOSS:{:.4f} LR:{:.8f}".format((epoch+1),
                                                                                       total_train_loss/(i+1), get_lr(optimizer)))

        # Calculate train metric
        metrics = ner_metrics_fn(list_hyp, list_label)
#         print("(Epoch {}) TRAIN LOSS:{:.4f} {} LR:{:.8f}".format((epoch+1),
#             total_train_loss/(i+1), metrics_to_string(metrics), get_lr(optimizer)))
        logger.info("(Epoch {}) TRAIN LOSS:{:.4f} {} LR:{:.8f}".format((epoch+1),
                                                                       total_train_loss/(i+1), metrics_to_string(metrics), get_lr(optimizer)))

        # Evaluate on validation
        model.eval()
        torch.set_grad_enabled(False)

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
#         print("(Epoch {}) VALID LOSS:{:.4f} {}".format((epoch+1),
#             total_loss/(i+1), metrics_to_string(metrics)))
        logger.info("(Epoch {}) VALID LOSS:{:.4f} {}".format((epoch+1),
                                                             total_loss/(i+1), metrics_to_string(metrics)))

#         if total_loss/(i+1) < min_loss and metrics["F1"] > max_f1 :
        if total_loss/(i+1) < min_loss:
            #             print("save model checkpoint")
            logger.info("save model checkpoint at {}".format(model_dir))

            min_loss = total_loss/(i+1)
            max_f1 = metrics["F1"]

            save("{}model-{}.pth".format(model_dir, epoch+1), model, optimizer)
            # https://github.com/huggingface/transformers/issues/7849

    df = pd.read_csv("data/test.csv")
    df["POI/street"] = df["raw_address"].apply(extract_poi_street)
    df = df.drop(columns=["raw_address"])

    SGT = pytz.timezone('Singapore')
    datetime_sgt = datetime.now(SGT)
    time_now = datetime_sgt.strftime('%Y-%m-%d--%H-%M-%S')

    csv_name = "submissions/bert-{}/{}.csv".format(model_version, time_now)
    logger.info("Submission saved at {}".format(csv_name))
    df.to_csv(csv_name, index=False)
