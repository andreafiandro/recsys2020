# RecSys2020 Challenge
# https://recsys-twitter.com/

# Sequence Classification with BERT for recsys2020 features
# Huggingface library
# https://huggingface.co/transformers/

# RCE Reverse Cross Entropy
# https://github.com/P2333/Reverse-Cross-Entropy

# Also taking inspiration from
# https://towardsdatascience.com/bert-classifier-just-another-pytorch-model-881b3cf05784
# https://github.com/sugi-chan/custom_bert_pipeline/blob/master/bert_pipeline.ipynb

# The dataset is available after login RecSys2020 Challenge at
# https://recsys-twitter.com/data/show-downloads 
# Some dataset preprocessing was performed by us, pay attention

# Add to run on colab notebook
#!pip3 install pytorch_pretrained_bert

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertConfig

from sklearn.model_selection import train_test_split
import pandas as pd
from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
from random import randrange

import logging # if you want to have more information on what's happening, activate the logger

#import torchvision
#from PIL import Image

logging.basicConfig(level=logging.INFO)
# Load pre-trained model tokenizer (vocabulary)
# Tokenization phase is not necessary
# because we have already the token IDS
# -> tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased') NOT NECESSARY

# Model for Bert Layer Normalization  

class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

# Model for BertForSequenceClassification       

class BertForSequenceClassification(nn.Module):
  
    def __init__(self, num_labels=2):
        super(BertForSequenceClassification, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        nn.init.xavier_normal_(self.classifier.weight)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
    
    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False
    
    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True

# Train model function - Return the trained model

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):

# Warning

# Prior to PyTorch 1.1.0, the learning rate scheduler was expected to be called before the optimizer’s update; 1.1.0 
# changed this behavior in a BC-breaking way. If you use the learning rate scheduler (calling scheduler.step()) before 
# the optimizer’s update (calling optimizer.step()), this will skip the first value of the learning rate schedule. 
# If you are unable to reproduce results after upgrading to PyTorch 1.1.0, please check if you are calling 
# scheduler.step() at the wrong time.

    since = time.time()
    print('starting')
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                #scheduler.step() -> DUE TO THE WARNING
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            
            output_corrects = 0
            
            
            # Iterate over data.
            for inputs, output in dataloaders_dict[phase]:
                #inputs = inputs
                print(len(inputs),type(inputs),inputs)
                #inputs = torch.from_numpy(np.array(inputs)).to(device) 
                inputs = inputs.to(device) 

                output = output.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    #print(inputs)
                    outputs = model(inputs)

                    outputs = F.softmax(outputs,dim=1)
                    
                    loss = criterion(outputs, torch.max(output.float(), 1)[1])
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        
                        loss.backward()
                        optimizer.step()
                        scheduler.step()  #-> DUE TO THE WARNING
                # statistics
                running_loss += loss.item() * inputs.size(0)

                
                output_corrects += torch.sum(torch.max(outputs, 1)[1] == torch.max(output, 1)[1])

                
            epoch_loss = running_loss / dataset_sizes[phase]

            
            output_acc = output_corrects.double() / dataset_sizes[phase]

            print('{} total loss: {:.4f} '.format(phase,epoch_loss ))
            print('{} output_acc: {:.4f}'.format(
                phase, output_acc))

            if phase == 'val' and epoch_loss < best_loss:
                print('saving with loss of {}'.format(epoch_loss),
                      'improved over previous {}'.format(best_loss))
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), 'bert_model_test.pth')


        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(float(best_loss)))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# Please, check if max sequence length is correct
# Team -JP- suggests that this number could be better chosen 512 -> 256 -> 190
# Reasoning over padding has to be done 

max_seq_length = 190

# Bring the dataset to a useful format

class text_dataset(Dataset):
    def __init__(self,x_y_list, transform=None):
        
        self.x_y_list = x_y_list
        self.transform = transform
        
    def __getitem__(self,index):
        
        # Tokenization phase is not necessary
        # because we have already the token IDS
        # -> tokenizer.tokenize  NOT NECESSARY
        # -> tokenizer.convert_tokens_to_ids NOT NECESSARY

        # -> tokenized_review = tokenizer.tokenize(self.x_y_list[0][index]) NOT NECESSARY
        # so:

        #print("Index element preprocessed")
        #print(index)
        #print("Element preprocessed")
        #print(self.x_y_list[0][index])

        tokenized_review = self.x_y_list[0][index]
        tokenized_review = tokenized_review.split("|")
        
        #print(tokenized_review)

        if len(tokenized_review) > max_seq_length:
            tokenized_review = tokenized_review[:max_seq_length]
            
        # -> ids_review  = tokenizer.convert_tokens_to_ids(tokenized_review) NOT NECESSARY
        # so:

        ids_review  = tokenized_review
        i = 0
        for x in ids_review:
          ids_review[i]=int(ids_review[i])
          i = i + 1

        padding = [0] * (max_seq_length - len(ids_review))
        
        ids_review += padding

        assert len(ids_review) == max_seq_length      
     
        ids_review = torch.tensor(ids_review)

        output = self.x_y_list[1][index] # color   
       
        list_of_labels = [torch.from_numpy(np.array(output))]

        return ids_review, list_of_labels[0]
    
    def __len__(self):
        return len(self.x_y_list[0])


#####################################################################################
"""MAIN START HERE"""
#####################################################################################

# Config Bert and BertForSequenceClassification Model

config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

# Here will change the number of labels and its format to obtain one or four model, and have as return value logits or CE loss + labels

num_labels = 2
model = BertForSequenceClassification(num_labels)

# Reading the dataset

# Open dataset training chunk and preview of content
nrows = 1024
batch_size = 64
dat = pd.read_csv('/content/drive/My Drive/training_chunk_0.csv',nrows=nrows)

print("DATASET SHAPE")
print(dat.shape)

print("HEAD FUNCTION")
print(dat.head())

# Fill the dataset NaN cells with 0, useful for preprocessing

dat = dat.fillna(0)

# Select the columns of our interest, text tokens and tweet type

# Reply engagement timestamp, Retweet engagement timestamp, Retweet with comment engagement timestamp, Like engagement timestamp

X = dat['Text_tokens']
y = dat['Like_engagement_timestamp'] # -> HAS TO BECOME A PARAMETER

# Split in train and test part the chunk

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# In this case, due the nature of text tokens field, we will have a list of string. 
# Each string is a sequence of token ids separed by | , that have to be correctly transformed into a list 
# (this will be done by text_data function).

# X_train = X_train.reset_index(drop=True)
# X_test = X_test.reset_index(drop=True)

X_train = X_train.values.tolist()
X_test = X_test.values.tolist()


# pandas.get_dummies
# Convert categorical variable into dummy/indicator variables.
# Examples
#
# s = pd.Series(list('abca'))
# pd.get_dummies(s)
#    a  b  c
# 0  1  0  0
# 1  0  1  0
# 2  0  0  1
# 3  1  0  0
# pd.get_dummies(s).values.tolist()
# [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]]
#
# ... it is like a one hot encoding for labels

# The labels are not enum but are represented in the dataset
# as empty cell if there wasn't no engagment or with a timestamp if was an engagment.
# The necessary transformations will take place through dat.fillna(0) and
# transformation lamda 

y_train = y_train.transform(lambda x: 1 if x>0 else 0)
y_test = y_test.transform(lambda x: 1 if x>0 else 0)

wg = y_train.value_counts()
print(wg)


print("y_train")

print(y_train)

y_train = pd.get_dummies(y_train).values.tolist()
y_test = pd.get_dummies(y_test).values.tolist()

print("dummies y_train")

print(y_train)

# Choose a batch size considering the chunk size



# Input preparation for Dataloader function

train_lists = [X_train, y_train]
test_lists = [X_test, y_test]

# Here, we perform the transformation needed over the input

training_dataset = text_dataset(x_y_list = train_lists )
test_dataset = text_dataset(x_y_list = test_lists )

dataloaders_dict = {'train': torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
                   'val':torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
                   }
dataset_sizes = {'train':len(train_lists[0]),
                'val':len(test_lists[0])}

# Choose GPU if is available, otherwise cpu

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Parameter for the training

lrlast = .001
lrmain = .00001
optim1 = optim.Adam(
    [
        {"params":model.bert.parameters(),"lr": lrmain},
        {"params":model.classifier.parameters(), "lr": lrlast},
       
   ])

# optim1 = optim.Adam(model.parameters(), lr=0.001)#,momentum=.9)
# Observe that all parameters are being optimized
optimizer_ft = optim1

# This criterion will be substitute with RCE Reverse Cross Entropy
# https://pytorch.org/docs/stable/nn.html#crossentropyloss
# Team -MS- recommends weighing the classes to compensate for the unbalanced dataset

weights = [nrows/ wg[0],nrows/(nrows-wg[0])]
print(weights)
class_weights = torch.FloatTensor(weights)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Decay LR by a factor of 0.1 every 7 epochs

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.1)

# Start Training

num_epochs = 10
model_ft1 = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,num_epochs)

