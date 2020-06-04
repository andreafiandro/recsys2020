import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from pytorch_pretrained_bert import BertTokenizer
import configSimo as c
import argparse
import os
import torch
from bert_model import BERT
from config import TestConfig, TrainingConfig
from recSysDataset import BertDatasetTest
import bert_test as bt5

#pathtomypers = "../../dataset/myPersonalitySmall/statuses_unicode.txt"
#pathtobig5scores = "../../dataset/myPersonalitySmall/big5labels.txt"
tokenizer = c.TOKENIZER

dfin = pd.read_csv("../input/sentiment/train.csv")
fout = open("sent_text_tokens.csv", "w")
sent = dfin["sentiment"]
dfin = dfin["text"]

counter = 0
for i in range(dfin.shape[0]):
    line = dfin[i]
    text = line.rstrip("\n")
    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    for token in indexed_tokens[:-1]:
        fout.write(str(token)+"|")
    fout.write(str(indexed_tokens[-1]) + "," + str(sent[i]) + "," + str(counter) + "," + str(15000+counter)+"\n")
    counter = counter + 1
fout.close()

#dataset = pd.read_csv("cls_table.csv", header=None) 


# fb5 = pd.read_csv(pd.read_csv(pathtobig5scores, delim_whitespace=True, header=None) 
# X = pd.read_csv("text_tokens.csv")

# bt5.cls_isa()
# y = Y.iloc[:,0] #working on Openness
# y = y.to_numpy()
# y = np.reshape(y, (len(y), 1))
# targets = torch.from_numpy(y)

print("the end")