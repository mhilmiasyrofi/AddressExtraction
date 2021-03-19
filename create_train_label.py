from collections import Counter 

import pandas as pd
import numpy as np
import re

def save_file(filename, arr) :
    f = open(filename, "w+",  encoding='utf-8')
    for s in arr :
        f.write(s.encode("ascii", "ignore").decode())
        f.write("\n")
    f.close()

def getPOI(s) :
    return s.split("/")[0]

def getStreet(s) :
    return s.split("/")[1]

def create_label(s, poi, street) :
    tokens = s.split(" ")
    n = len(tokens)
    tp = poi.split(" ")
    ts = street.split(" ")
    tags = []
    i = 0
    while i < n :
        if len(tp) > 0 and tokens[i] == tp[0] :
            j = 1
            while j < len(tp) :
                if i+j < n and tokens[i+j] == tp[j]:
                    j += 1
                else :
                    break
            if j == len(tp) :
                tags.append("B-POI")
                for k in range(1, j) :
                    tags.append("I-POI")
                i += j
            else :
                tags.append("O")
                i += 1
        elif len(ts) > 0 and tokens[i] == ts[0] :
            j = 1
            while j < len(ts) :
                if i+j < n and tokens[i+j] == ts[j] :
                    j += 1
                else :
                    break
            if j == len(ts) :
                tags.append("B-STREET")
                for k in range(1, j) :
                    tags.append("I-STREET")
                i += j
            else :
                tags.append("O")
                i += 1                
        else :
            tags.append("O")
            i += 1
        
#         print(tokens)
#         print(tags)

    if tokens[-1] != "." :
        tokens.append(".")
        tags.append("O")
    
    res = ""
#     print(tokens)
#     print(tags)
    for token, tag in zip(tokens, tags) :
        res += "{}\t{}\n".format(token, tag)
    return res

def test() :
    s = "raya samb gede , 299 toko bb kids"
    poi = "toko bb kids"
    street = "raya samb gede"
    res = create_label(s, poi, street)
    print(res)

if __name__ == "__main__" :
    data_path = "data/"
    df = pd.read_csv(data_path + "cleaned_train.csv")    

    df["POI"] = df["POI/street"].apply(getPOI)
    df["Street"] = df["POI/street"].apply(getStreet) 
    df["label"] = df.apply(lambda x: create_label(x.raw_address, x.POI, x.Street), axis=1)

    final = df.sample(frac=1, random_state=0).reset_index(drop=True)
    n = len(final)
    nt = int(0.95 * n)
    train = final[:nt+1]
    test = final[nt:]

    train_path = data_path + "bert-fine-tune/train.txt"
    validation_path = data_path + "bert-fine-tune/validation.txt"

    train_data = train["label"].values.tolist()
    validation_data = test["label"].values.tolist()

    ntry = None
    # ntry = 1000
    if ntry :
        train_data = train_data[:ntry]
        validation_data = validation_data[:ntry]

    save_file(train_path, train_data)
    save_file(validation_path, validation_data)





