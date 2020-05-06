import argparse
import math
import os
import pdb
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from bert_model import BERT
from config import TrainingConfig
from recSysDataset import BertDatasetTest
from transformers import BertForSequenceClassification


_PRINT_INTERMEDIATE_LOG = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def preprocessing(df, args):

    df = df.fillna(0)
    text = df[args.tokcolumn]
    tweetid = df[args.tweetidcolumn]
    user = df[args.usercolumn] 

    text_test = text.values.tolist()
    tweetid_test = tweetid.values.tolist()
    user_test = user.values.tolist()

    return text_test,tweetid_test,user_test

def main():

    parser = argparse.ArgumentParser()

    #read user parameters
    parser.add_argument(
        "--data",
        default=None,
        type=str,
        required=True,
        help="Path to dataset"
    )
    parser.add_argument(
        "--tokcolumn",
        default=None,
        type=str,
        required=True,
        help="Column name for bert tokens (e.g. \"text tokens\")"
    )
    parser.add_argument(
        "--tweetidcolumn",
        default=None,
        type=str,
        required=True,
        help="Column name for tweet id (e.g. \"tweet id\")"
    )
    parser.add_argument(
        "--usercolumn",
        default=None,
        type=str,
        required=True,
        help="Column name for user id (e.g. \"user id\")"
    )

    
    args = parser.parse_args()

    # Initializing a BERT model
    model = BERT(pretrained=TrainingConfig._pretrained_bert, n_labels=TrainingConfig._num_labels, dropout_prob = TrainingConfig._dropout_prob, freeze_bert = True)
    
    # load del nostro modello per la fase di test

    checkpoint = torch.load(os.path.join(TrainingConfig._checkpoint_path, 'bert_model_test.pth'))
    model.load_state_dict(checkpoint)
    model.eval()

    if _PRINT_INTERMEDIATE_LOG:
        print(model.config)

    df = pd.read_csv(args.data)

    if _PRINT_INTERMEDIATE_LOG:
        print('DATASET SHAPE: '+ str(df.shape))
        print('HEAD FUNCTION: '+ str(df.head()))

    # select and preprocess columns <Tweet_Id>,<User_Id>

    text_test_chunk, tweetid_test_chunk, user_test_chunk = preprocessing(df, args)
    print("Number of training rows "+str(len(text_test_chunk)))

    # create the dataset objects
    
    test_data = BertDatasetTest(xy_list=[text_test_chunk,tweetid_test_chunk,user_test_chunk])
    test_data = torch.utils.data.DataLoader(test_data)

    # move model to device before optimizer instantiation
    model.to(device)
    submission_dataframe = pd.DataFrame(columns=['Tweet_Id', 'User_Id','Reply','Retweet','Retweet_with_comment','Like'])

    i=0
    for line in test_data:
        text_line = torch.from_numpy(np.array(line[0])).to(device)
        logits, cls_output = model(text_line)
        logits = [ (lambda x: logit2prob(x))(x) for x in logits[0].tolist()]

        
        submission_dataframe.loc[i] = [line[1][0],line[2][0]]+logits
        i = i + 1

    if _PRINT_INTERMEDIATE_LOG:
        print('DATASET SHAPE: '+ str(submission_dataframe.shape))
        print('HEAD FUNCTION: '+ str(submission_dataframe.head()))
    
    # make submission file <Tweet_Id>,<User_Id>,<Prediction>
    submission_dataframe.rename(columns={'Reply':'Prediction'}, inplace=True)
    submission_dataframe.to_csv(r'reply_prediction.csv',columns = ['Tweet_Id','User_Id','Prediction'],index=False)
    submission_dataframe.rename(columns={'Prediction':'Reply'}, inplace=True)
    submission_dataframe.rename(columns={'Retweet':'Prediction'}, inplace=True)
    submission_dataframe.to_csv(r'retweet_prediction.csv',columns = ['Tweet_Id','User_Id','Prediction'],index=False)
    submission_dataframe.rename(columns={'Prediction':'Retweet'}, inplace=True)
    submission_dataframe.rename(columns={'Retweet_with_comment':'Prediction'}, inplace=True)
    submission_dataframe.to_csv(r'retweet_with_comment_prediction.csv',columns = ['Tweet_Id','User_Id','Prediction'],index=False)
    submission_dataframe.rename(columns={'Prediction':'Retweet_with_comment'}, inplace=True)
    submission_dataframe.rename(columns={'Like':'Prediction'}, inplace=True)
    submission_dataframe.to_csv(r'like_prediction.csv',columns = ['Tweet_Id','User_Id','Prediction'],index=False)
    submission_dataframe.rename(columns={'Prediction':'Like'}, inplace=True)
    
def logit2prob(logit):
  odds = math.exp(logit)
  prob = odds / (1 + odds)
  return prob

    
if __name__ == "__main__":
    main()