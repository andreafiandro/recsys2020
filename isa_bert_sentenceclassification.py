# DON'T RUN, IT IS A DRAFT

# Sequence Classification with BERT for recsys2020 features
# Huggingface library
# https://huggingface.co/transformers/
# Also taking ispiration from
# https://towardsdatascience.com/bert-classifier-just-another-pytorch-model-881b3cf05784
# https://github.com/sugi-chan/custom_bert_pipeline/blob/master/bert_pipeline.ipynb
# Dataset Folder
# https://istitutoboella-my.sharepoint.com/:f:/g/personal/giuseppe_rizzo_linksfoundation_com/EibesId87KJIrUJ252lS_CQBsv0hPG0T-O1bortw4zTIhQ?e=5%3aByodq4&at=9 


import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertConfig
import logging # Optional: if you want to have more information on what's happening, activate the logger as follows
from __future__ import print_function, division
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from random import randrange
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import pandas as pd


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
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            
            output_corrects = 0
            
            
            # Iterate over data.
            for inputs, output in dataloaders_dict[phase]:
                #inputs = inputs
                #print(len(inputs),type(inputs),inputs)
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

                # statistics
                running_loss += loss.item() * inputs.size(0)

                
                output_corrects += torch.sum(torch.max(outputs, 1)[1] == torch.max(output, 1)[1])

                
            epoch_loss = running_loss / dataset_sizes[phase]

            
            output_acc = output_corrects.double() / dataset_sizes[phase]

            print('{} total loss: {:.4f} '.format(phase,epoch_loss ))
            print('{} output_acc: {:.4f}'.format(
                phase, sentiment_acc))

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

max_seq_length = 512

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
        tokenized_review = self.x_y_list[0][index]

        if len(tokenized_review) > max_seq_length:
            tokenized_review = tokenized_review[:max_seq_length]
            
        # -> ids_review  = tokenizer.convert_tokens_to_ids(tokenized_review) NOT NECESSARY
        # so:

        ids_review  = tokenized_review

        padding = [0] * (max_seq_length - len(ids_review))
        
        ids_review += padding
        
        assert len(ids_review) == max_seq_length
        
        print(ids_review)
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

dat = pd.read_csv('isa_puppy_chunk_recsys2020.csv')
dat.head()

# Select the columns of our interest, text tokens and tweet type

X = dat['Text tokens']
y = dat['Tweet type']

# Split in train and test part the chunk

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# In this case, due the nature of text tokens field, we will have a list of string. 
# Each string is a sequence of token ids separed by | , that have to be correctly transformed into a list 
# (this will be done by text_data function).

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

y_train = pd.get_dummies(y_train).values.tolist()
y_test = pd.get_dummies(y_test).values.tolist()

# Choose a batch size considering the chunck size

batch_size = 16

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
criterion = nn.CrossEntropyLoss()

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.1)

model_ft1 = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=10)


####################################################
"""COSE CHE NON SO DOVE METTERE, UN ATTIMO """
####################################################

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor(list_sequences_token_ids) # forse questo non serve perchè era solo un esempio
logits = model(tokens_tensor) # forse questo non serve perchè era solo un esempio

# COMM import torch.nn.functional as F

F.softmax(logits,dim=1)  #questo sicuro da qualche parte lo devo mettere 


