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
from config import TestConfig
from config import TrainingConfig
from recSysDataset import BertDatasetTest
from nlprecsysutility import submission_files
from transformers import BertForSequenceClassification


_PRINT_INTERMEDIATE_LOG = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
os.makedirs(TestConfig._output_dir, exist_ok=True)

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
    parser.add_argument(
        "--batch",
        default=None,
        type=int,
        required=True,
        help="Batch size for the training"
    )
    parser.add_argument(
        "--workers",
        default=None,
        type=int,
        required=True,
        help="Number of workers for the training"
    )

    
    args = parser.parse_args()

    # Initializing a BERT model
    model = BERT(pretrained=TrainingConfig._pretrained_bert, n_labels=TrainingConfig._num_labels, dropout_prob = TrainingConfig._dropout_prob, freeze_bert = True)
    
    # load del nostro modello per la fase di test,  andrebbe caricato questo direttamente al posto del pretrained bert

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
    test_data = torch.utils.data.DataLoader(test_data,batch_size=args.batch, shuffle=True, num_workers=args.workers)

    # move model to device
    model.to(device)

    # test dataframe 
    columns = ['Tweet_Id', 'User_Id','Reply','Retweet','Retweet_with_comment','Like']
    submission_dataframe = pd.DataFrame(columns=columns)

    # eval per batch, controllare corrispondenza colonne

    m = nn.Sigmoid()
    for lines in test_data:
        # eval 
        text_lines = torch.from_numpy(np.array(lines[0])).to(device)
        logits, cls_output = model(text_lines)

        # from logits to probability
        logits = m(logits)

        # traspongo i logit per inserirli corettamente nel dataframe
        # così come sono i dati ora sono trasposti rispetto a come li prende dataframe
        logits = np.array(logits.data.tolist()).T 

        # dict dati per dataframe, sicuramente si può fare in maniera più bella
        batch_data = {'Tweet_Id':list(lines[1]),'User_Id':list(lines[2]),'Reply':logits[0],'Retweet':logits[1],'Retweet_with_comment':logits[2],'Like':logits[3]}

        print(batch_data)
        submission_dataframe= submission_dataframe.append(pd.DataFrame(batch_data,columns=columns))

    if _PRINT_INTERMEDIATE_LOG:
        print('DATASET SHAPE: '+ str(submission_dataframe.shape))
        print('HEAD FUNCTION: '+ str(submission_dataframe.head()))
    
    # make submission file <Tweet_Id>,<User_Id>,<Prediction>

    submission_files(submission_dataframe, output_dir = TestConfig._output_dir)

# from logit to prob  

def logit2prob(logit):
  odds = math.exp(logit)
  prob = odds / (1 + odds)
  return prob

    
if __name__ == "__main__":
    main()