from collections import Counter

import pandas as pd
import numpy as np
import re


def preprocess(s):
    s = s.replace("\"", "")
    s = s.replace("\t", " ")
    s = re.sub(r'([,/\\]+)', r' \1 ', s)
    s = re.sub(' +', ' ', s)  # change multiple spaces into one space
    return s.strip()


def create_label(s, poi, street):
    tokens = s.split(" ")
    n = len(tokens)
    tp = poi.split(" ")
    ts = street.split(" ")
    tags = []
    i = 0
    while i < n:
        if len(tp) > 0 and tokens[i] == tp[0]:
            j = 1
            while j < len(tp):
                if i+j < n and tokens[i+j] == tp[j]:
                    j += 1
                else:
                    break
            if j == len(tp):
                tags.append("B-ORGANISATION")
                for k in range(1, j):
                    tags.append("I-ORGANISATION")
                i += j
            else:
                tags.append("O")
                i += 1
        elif len(ts) > 0 and tokens[i] == ts[0]:
            j = 1
            while j < len(ts):
                if i+j < n and tokens[i+j] == ts[j]:
                    j += 1
                else:
                    break
            if j == len(ts):
                tags.append("B-PLACE")
                for k in range(1, j):
                    tags.append("I-PLACE")
                i += j
            else:
                tags.append("O")
                i += 1
        else:
            tags.append("O")
            i += 1

#         print(tokens)
#         print(tags)

    if tokens[-1] != ".":
        tokens.append(".")
        tags.append("O")

    res = ""
#     print(tokens)
#     print(tags)
    for token, tag in zip(tokens, tags):
        res += "{}\t{}\n".format(token, tag)
    return res


def sanity_check(df):
    for sentence, labels, poi, street in zip(df["raw_address"], df["Label"], df["POI"], df["Street"]):
        for l in labels.split("\n"):
            if len(l.split("\t")) > 2:
                print("Sentence:", sentence)
                print("POI:", poi)
                print("Street:", street)
                print("Bad Label: ", l)
                return


def save_file(filename, arr):
    f = open(filename, "w+",  encoding='utf-8')
    for s in arr:
        f.write(s.encode("ascii", "ignore").decode())
        f.write("\n")
    f.close()


if __name__ == "__main__":
    df = pd.read_csv("cleaned_train.csv")
    # df = pd.read_csv("train.csv")

    df.rename(columns={"street": "Street"}, inplace=True)
    df = df.replace(np.nan, "")

    df["raw_address"] = df["raw_address"].apply(preprocess)

    df["Label"] = df.apply(lambda x: create_label(
        x.raw_address, x.POI, x.Street), axis=1)

    sanity_check(df)

    final = df.sample(frac=1, random_state=0).reset_index(drop=True)
#     final = final[:200000]

    n = len(final)
    nt = int(0.6 * n)
    train = final[:nt+1]
    test = final[nt:]

    ntry = None
    # ntry = 1000
    if ntry:
        save_file("train_preprocess.txt",
                  train["Label"].values.tolist()[:ntry])
        save_file("valid_preprocess.txt", test["Label"].values.tolist()[:ntry])
    else:
        save_file("train_preprocess.txt", train["Label"].values.tolist())
        save_file("valid_preprocess.txt", test["Label"].values.tolist())
