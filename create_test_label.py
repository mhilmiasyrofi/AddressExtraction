from collections import Counter 

import pandas as pd
import numpy as np
import re

def create_test_label(s) :
    tokens = s.split(" ")
    n = len(tokens)
    tags = []
    i = 0
    while i < n :
        tags.append("O")
        i += 1

    if tokens[-1] != "." :
        tokens.append(".")
        tags.append("O")
    
    res = ""
#     print(tokens)
#     print(tags)
    for token, tag in zip(tokens, tags) :
        res += "{}\t{}\n".format(token, tag)
    return res

def save_file(filename, arr) :
    f = open(filename, "w+",  encoding='utf-8')
    for s in arr :
        f.write(s.encode("ascii", "ignore").decode())
        f.write("\n")
    f.close()

def test() :
    s = "raya samb gede , 299 toko bb kids"
    s = "s. par 53 sidanegara 4 cilacap tengah"
    s = "adi ,"
    res = create_test_label(s)
    print(res)

if __name__ == "__main__" :
    data_path = "data/"
    df = pd.read_csv(data_path + "cleaned_test.csv")
    df["label"] = df.apply(lambda x: create_test_label(x.raw_address), axis=1)
    test_path = data_path + "bert-fine-tune/test.txt"
    test_data = df["label"].values.tolist()
    save_file(test_path, test_data)

