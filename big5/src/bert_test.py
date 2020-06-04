import argparse
import os
import numpy as np
import pandas as pd
import torch
from bert_model import BERT
from config import TestConfig, TrainingConfig
from recSysDataset import BertDatasetTest


def preprocessing(df, args):
    df = df.fillna(0)
    text = df[args.tokcolumn]
    tweetid = df[args.tweetidcolumn]
    user = df[args.usercolumn]
    text_test = text.values.tolist()
    tweetid_test = tweetid.values.tolist()
    user_test = user.values.tolist()
    return text_test, tweetid_test, user_test


def cls_isa():
    _PRINT_INTERMEDIATE_LOG = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(device)
    os.makedirs(TestConfig._output_dir, exist_ok=True)
    parser = argparse.ArgumentParser()

    # read user parameters
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

    model = BERT(pretrained=TrainingConfig._pretrained_bert,
                 n_labels=TrainingConfig._num_labels,
                 dropout_prob=TrainingConfig._dropout_prob,
                 freeze_bert=True)

    if _PRINT_INTERMEDIATE_LOG:
        print(model.config)

    df = pd.read_csv(args.data)

    df = df.iloc[0:100000]  #### introdotto per la fase di debugging

    if _PRINT_INTERMEDIATE_LOG:
        print('DATASET SHAPE: ' + str(df.shape))
        print('HEAD FUNCTION: ' + str(df.head()))

    # select and preprocess columns <Tweet_Id>,<User_Id>
    text_test_chunk, \
        tweetid_test_chunk, user_test_chunk = preprocessing(df, args)
    if _PRINT_INTERMEDIATE_LOG:
        print("Number of rows "+str(len(text_test_chunk)))

    # create the dataset objects
    test_data = BertDatasetTest(xy_list=[text_test_chunk,
                                         tweetid_test_chunk,
                                         user_test_chunk])
    test_data = torch.utils.data.DataLoader(test_data,
                                            batch_size=args.batch,
                                            shuffle=False,
                                            num_workers=args.workers)
    # model.to(device)
    # fout = open("csv_table.csv","w")
    flag = 0
    iteration = 1
    for lines in test_data:
        print("iter", iteration, " out of ", str(df.shape[0]/args.batch))
        iteration = iteration + 1
        text_lines = torch.from_numpy(np.array(lines[0]))  # .to(device)
        logits, cls_output = model(text_lines)
        # print(cls_output.numpy())
        if flag == 0:
            x = cls_output
            flag = 1
        else:
            x = torch.cat((x, cls_output), dim=0)
    return x
