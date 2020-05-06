from torch.utils.data import Dataset
import torch
import numpy as np

import pdb


class BertDataset(Dataset):
    """Dataset class for Bert data
    """
    

    def __init__(self, xy_list):

        self.xy_list = [xy_list[0], xy_list[1], xy_list[2], xy_list[3], xy_list[4]]
  
        ########################################
        # TODO
        # Please, check if max sequence length is correct
        # Team -JP- suggests that this number could be better chosen 512 -> 256 -> 150
        # Reasoning over padding has to be done
        ########################################
        self.max_seq_len = 150


    def __getitem__(self, index):
        ########################################
        # Tokenization phase is not necessary
        # because we have already the token IDS
        # -> tokenizer.tokenize  NOT NECESSARY
        # -> tokenizer.convert_tokens_to_ids NOT NECESSARY
        ########################################
        tokenized_review = self.xy_list[0][index]
        tokenized_review = tokenized_review.split("|")
        
        if len(tokenized_review) > self.max_seq_len:
            tokenized_review = tokenized_review[:self.max_seq_len]


        ids_review  = tokenized_review
        i = 0
        for x in ids_review:
          ids_review[i]=int(ids_review[i])
          i = i + 1

        padding = [0] * (self.max_seq_len - len(ids_review))
        
        ids_review += padding

        assert len(ids_review) == self.max_seq_len      
     
        ids_review = torch.tensor(ids_review)

        #target = self.xy_list[1][index] # single label
        target = [self.xy_list[1][index],self.xy_list[2][index],self.xy_list[3][index],self.xy_list[4][index]]

        target = torch.tensor(target, dtype=torch.int64)

        return ids_review, target


    def __len__(self):
        return len(self.xy_list[0])
        

class BertDatasetTest(Dataset):
    """Dataset class for Bert data
    """
    

    def __init__(self, xy_list):

        self.xy_list = xy_list
  
        ########################################
        # TODO
        # Please, check if max sequence length is correct
        # Team -JP- suggests that this number could be better chosen 512 -> 256 -> 150
        # Reasoning over padding has to be done
        ########################################
        self.max_seq_len = 150


    def __getitem__(self, index):
        ########################################
        # Tokenization phase is not necessary
        # because we have already the token IDS
        # -> tokenizer.tokenize  NOT NECESSARY
        # -> tokenizer.convert_tokens_to_ids NOT NECESSARY
        ########################################
        
        tokenized_review = self.xy_list[0][index]
        
        tokenized_review = tokenized_review.split("|")
        
        if len(tokenized_review) > self.max_seq_len:
            tokenized_review = tokenized_review[:self.max_seq_len]


        ids_review  = tokenized_review
        i = 0
        for x in ids_review:
          ids_review[i]=int(ids_review[i])
          i = i + 1

        padding = [0] * (self.max_seq_len - len(ids_review))
        
        ids_review += padding

        assert len(ids_review) == self.max_seq_len      
     
        ids_review = torch.tensor(ids_review)

        tweet_id = self.xy_list[1][index]
        user_id = self.xy_list[2][index]

        return ids_review, tweet_id, user_id


    def __len__(self):
        return len(self.xy_list[0])
        