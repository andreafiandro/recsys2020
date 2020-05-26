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
from recSysDataset import BertDatasetTest, CNN_Features_Dataset_Test
from nlprecsysutility import submission_files, RecSysUtility
from transformers import BertForSequenceClassification

from cnn_model import TEXT_ENSEMBLE, CNN, FEATURES_ENSEMBLE


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

def prepro_features(df, args):
    df = df.fillna(0)
    text = df[args.tokcolumn]
    tweetid = df[args.tweetidcolumn]
    user = df[args.usercolumn] 
    
    dummy = RecSysUtility('')
    feats = dummy.generate_features_lgb_mod(df, user_features_file=args.ufeatspath) #Note: Slithly different from other branch this returns text_tokens column
    feats = dummy.encode_val_string_features(feats)
    not_useful_cols = [args.tokcolumn, args.tweetidcolumn, 'User_id', args.usercolumn]
    feats.drop(not_useful_cols, axis=1, inplace=True)
    for col in feats.columns[:]:
        x[col] = feats[col].astype(float)
    
    text_test = text.values.tolist()
    tweetid_test = tweetid.values.tolist()
    user_test = user.values.tolist()
    feats_test = feats.values.tolist()

    return text_test, tweetid_test, user_test, feats_test

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
        default='Text_tokens',
        type=str,
        required=False,
        help="Column name for bert tokens (e.g. \"text tokens\")"
    )
    parser.add_argument(
        "--tweetidcolumn",
        default='Tweet_id',
        type=str,
        required=False,
        help="Column name for tweet id (e.g. \"Tweet_id\")"
    )
    parser.add_argument(
        "--usercolumn",
        default='User_id_engaging',
        type=str,
        required=False,
        help="Column name for user id target (e.g. \"User_id_engaging\")"
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
        default=2,
        type=int,
        required=False,
        help="Number of workers for the training"
    )
    parser.add_argument(
        '--features',
        default=None,
        type=str,
        required=False,
        help='Type \'--features yes\' to test the cnn+features model'
    )
    parser.add_argument(
        '--ufeatspath',
        default='./checkpoint/user_features_final.csv',
        type=str,
        required=False,
        help='Path to user_features.csv. Default=\'./checkpoint/user_features_final.csv\''
    )

    nrows = None
    args = parser.parse_args()
    # Initializing a BERT model
    bert_model = BERT(pretrained=TrainingConfig._pretrained_bert, n_labels=TrainingConfig._num_labels, dropout_prob = TrainingConfig._dropout_prob, freeze_bert = True)
    if args.features:
        cnn = CNN(dim=798, length=21) #768 cls +30 features = 768 % 21 => batch_sizex21x38
    else:
        cnn = CNN()

    model = FEATURES_ENSEMBLE(bert = bert_model, model_b = cnn)
    # load del nostro modello per la fase di test,  andrebbe caricato questo direttamente al posto del pretrained bert

    checkpoint = torch.load(os.path.join(TrainingConfig._checkpoint_path, 'cnn_model_test.pth'))
    model.load_state_dict(checkpoint)
    model.eval()

    if _PRINT_INTERMEDIATE_LOG:
        print(model.config)

    df = pd.read_csv(args.data, nrows= nrows)

    if _PRINT_INTERMEDIATE_LOG:
        print('DATASET SHAPE: '+ str(df.shape))
        print('HEAD FUNCTION: '+ str(df.head()))
    # select and preprocess columns <Tweet_Id>,<User_Id>
    if args.features:
        text_test_chunk, tweetid_test_chunk, user_test_chunk, feats_test_chunk = prepro_features(df, args)
        test_data = CNN_Features_Dataset_Test(text_test_chunk, tweetid_test_chunk, user_test_chunk, feats_test_chunk)
    else:
        text_test_chunk, tweetid_test_chunk, user_test_chunk = preprocessing(df, args)
        # create the dataset objects
        test_data = BertDatasetTest(xy_list=[text_test_chunk,tweetid_test_chunk,user_test_chunk])
    
    test_data = torch.utils.data.DataLoader(test_data, batch_size=args.batch, shuffle=False, num_workers=args.workers)

    if _PRINT_INTERMEDIATE_LOG:
        print("Number of training rows "+str(len(test_data)))
    # move model to device

    model.to(device)

    # test dataframe 

    columns = ['Tweet_Id', 'User_Id','Reply','Retweet','Retweet_with_comment','Like']
    submission_dataframe = pd.DataFrame(columns=columns)

    # eval per batch, controllare corrispondenza colonne

    #m = nn.Sigmoid() - Deprecated
    for lines in test_data:
        # eval 
        text_lines = torch.from_numpy(np.array(lines[0])).to(device)
        if args.features:
            feats = torch.from_numpy(np.array(lines[3])).to(device)
            logits, _ = model(text_lines, feats)
        else:
            logits, _ = model(text_lines)

        # from logits to probability
        logits = torch.sigmoid(logits)

        # traspongo i logit per inserirli corettamente nel dataframe
        # così come sono i dati ora sono trasposti rispetto a come li prende dataframe

        logits = np.array(logits.data.tolist()).T 

        # dict dati per dataframe, sicuramente si può fare in maniera più bella
        # Probably not a meno di usare dask al posto di pandas
        batch_data = {'Tweet_Id':list(lines[1]),'User_Id':list(lines[2]),'Reply':logits[0],'Retweet':logits[1],'Retweet_with_comment':logits[2],'Like':logits[3]}
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
