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

from utils.data_utils import NerShopeeDataset, NerDataLoader
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


def save(fpath, model, optimizer):
    # save
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, fpath)


if __name__ == "__main__":

    # Set random seed
    set_seed(26092020)

    model_version = "base"
    # model_version = "large"

    model_dir = "models/bert-{}/".format(model_version)
    
    
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

    model_dir = "{}{}_{}_{}/".format(model_dir, batch_size, max_seq_len, learning_rate)

    # Train
    n_epochs = 2
    if model_version == "large":
        n_epochs = 2

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
    config.num_labels = NerShopeeDataset.NUM_LABELS

    # Instantiate model
    model = BertForWordClassification.from_pretrained(
        model_name, config=config)

    train_dataset_path = 'data/bert-fine-tune/train.txt'
    valid_dataset_path = 'data/bert-fine-tune/validation.txt'

    train_dataset = NerShopeeDataset(
        train_dataset_path, tokenizer, lowercase=True)
    valid_dataset = NerShopeeDataset(
        valid_dataset_path, tokenizer, lowercase=True)


    train_loader = NerDataLoader(dataset=train_dataset, max_seq_len=max_seq_len,
                                 batch_size=batch_size, num_workers=16, shuffle=True)
    valid_loader = NerDataLoader(dataset=valid_dataset, max_seq_len=max_seq_len,
                                 batch_size=eval_batch_size, num_workers=16, shuffle=False)

    w2i, i2w = NerShopeeDataset.LABEL2INDEX, NerShopeeDataset.INDEX2LABEL

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]

#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.Adam(optimizer_grouped_parameters, lr=learning_rate)

    model = model.cuda()

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

            tr_loss = loss.item()
            
            if np.isnan(tr_loss) :
                break
            else :
                # Update model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

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
        
        save("{}model-{}.pth".format(model_dir, epoch+1), model, optimizer)
        logger.info("save model checkpoint at {}".format(model_dir))

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

        if total_loss/(i+1) < min_loss and metrics["F1"] > max_f1 :
#         if total_loss/(i+1) < min_loss:
            #             print("save model checkpoint")
            logger.info("save model checkpoint at {}".format(model_dir))

            min_loss = total_loss/(i+1)
            max_f1 = metrics["F1"]

            save("{}model-best.pth".format(model_dir), model, optimizer)
            # https://github.com/huggingface/transformers/issues/7849
