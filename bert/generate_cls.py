import argparse
import math
import os, gc
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
from nlprecsysutility import submission_files, RecSysUtility
from transformers import BertForSequenceClassification
from torch.utils.data import Dataset


_PRINT_INTERMEDIATE_LOG = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
os.makedirs(TestConfig._output_dir, exist_ok=True)

def merge_df_with_bz2(df, folder_path = './', tokcolumn='Text_tokens'):
    """
    Function to merge the input df with all the df present in folder path saved terminating with '.bz2' in binary with compression bz2
    :param df: Dataframe with tokcolumn to merge
    :param folder_path: path of the folder containing .bz2 files
    :param tokcolumn: column name of text_tokens eg. Text_tokens
    """
    if df is None:
        return df
    for file in os.listdir(folder_path):
        if file.endswith('.bz2'):
            print('Elaborating: %s' %file)
            df_tmp = pd.read_pickle(os.path.join(folder_path,file), compression='bz2')
            df = df.merge(df_tmp, how='left', left_on=tokcolumn, right_on=tokcolumn)
            del df_tmp
            gc.collect()
    return df

class TokensDatasetTest(Dataset):
    """Dataset class for Bert data only tokens
    """
    def __init__(self, tokens):
        self.tokens = tokens
        self.max_seq_len = 150


    def __getitem__(self, index):
        ########################################
        # Tokenization phase is not necessary
        # because we have already the token IDS
        # -> tokenizer.tokenize  NOT NECESSARY
        # -> tokenizer.convert_tokens_to_ids NOT NECESSARY
        ########################################
        tokenized_review = self.tokens[index][0]
        tokenized_review = tokenized_review.split("|")
        if len(tokenized_review) > self.max_seq_len:
            tokenized_review = tokenized_review[:self.max_seq_len]

        ids_review  = tokenized_review
        i = 0
        for _ in ids_review:
            ids_review[i]=int(ids_review[i])
            i = i + 1

        padding = [0] * (self.max_seq_len - len(ids_review))
        ids_review += padding
        assert len(ids_review) == self.max_seq_len      
        ids_review = torch.tensor(ids_review)

        return ids_review, self.tokens[index][0]


    def __len__(self):
        return len(self.tokens)

class CLSDatasetTest(Dataset):
    """Dataset class for Bert data only tokens
    """
    def __init__(self, cls_list):
        self.data = cls_list


    def __getitem__(self, index):
        cls = self.data[index]
        cls = torch.from_numpy(np.array(cls, dtype=np.float32))#.float()
        return cls


    def __len__(self):
        return len(self.data)

class test_model(nn.Module):
    def __init__(self):
        """
        
        """
        super(test_model, self).__init__()
        #Fill
    
    def forward(self, inputs):
        print(inputs.shape)
        return

def compute_cls(data, model, args, output_file='./tok_cls.csv'):
    test_data = TokensDatasetTest(data.values.tolist())
    test_data = torch.utils.data.DataLoader(test_data, batch_size=args.batch, shuffle=False, num_workers=args.workers)
    # move model to device
    model.to(device)
    # test dataframe 
    columns=[args.tokcolumn,'cls']
    generated = pd.DataFrame(columns=columns)
    print('Generating cls of %d' %len(data.index))

    for lines in test_data:
        text_lines = torch.from_numpy(np.array(lines[0])).to(device)
        _, cls = model(text_lines)
        cls = cls.cpu().detach().numpy()
        cls[cls>np.finfo(np.float16).max] = np.finfo(np.float16).max
        cls[cls<np.finfo(np.float16).min] = np.finfo(np.float16).min
        cls = cls.astype(np.float16)
        batch_data = {args.tokcolumn: list(lines[1]),'cls':list(cls)}
        generated= generated.append(pd.DataFrame(batch_data, columns=columns))

    print('Generated:', generated.shape,'\n', generated.head())
    print('Writing to %s...' %output_file)
    generated.to_pickle(output_file, compression='bz2')
    """
    generated.to_csv(output_file, header=True, index=False)
    generated[args.tokcolumn].to_csv(output_file+'t.csv', header=True, index=False)
    generated['cls'].to_csv(output_file+'c.csv', header=True, index=False)
    """
    print('Done.')
    """
    print('data',data.head())
    d2 = data.merge(generated, how='left', left_on=args.tokcolumn, right_on=args.tokcolumn)
    print('merge', d2.head(), d2.columns)
    t2 = CLSDatasetTest(d2['cls'].values.tolist())
    t2 = torch.utils.data.DataLoader(t2, batch_size=3, shuffle=False, num_workers=args.workers)
    l2 = next(iter(t2))
    test_data = TokensDatasetTest(data.values.tolist())
    test_data = torch.utils.data.DataLoader(test_data, batch_size=3, shuffle=False, num_workers=args.workers)
    for lines in test_data:
        #print(lines[1][0])
        #print(generated.loc[generated[args.tokcolumn]==lines[1][0]])
        print(generated.loc[generated[args.tokcolumn]==lines[1][0]]['cls'])
        text_lines = torch.from_numpy(np.array(lines[0])).to(device)
        _, cls = model(text_lines)
        print(cls)
        #print(np.equal(generated.loc[generated[args.tokcolumn]==lines[1][0]]['cls'][0].astype(np.float),cls.cpu().numpy()))
        #input()
        yt = l2.to(device)
        print(yt)
        print(cls.shape, yt.shape)
        print(torch.equal(cls,yt))
        input()
        l2 = next(iter(t2))
    input()
    if _PRINT_INTERMEDIATE_LOG:
        print('DATASET SHAPE: '+ str(generated.shape))
        print('HEAD FUNCTION: '+ str(generated.head()))
    """

def main():

    parser = argparse.ArgumentParser()

    #read user parameters
    parser.add_argument(
        "--data",
        default=None,
        type=str,
        required=False,
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
        "--batch",
        default=1000,
        type=int,
        required=False,
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
        "--jobn",
        default=None,
        type=int,
        required=False,
        help="Job_id if data is to split read across multiple jobs"
    )
    parser.add_argument(
        "--unique",
        default=None,
        type=str,
        required=False,
        help="Generate a unique Text_tokens csv from a tsv"
    )
    parser.add_argument(
        "--device",
        default=None,
        type=str,
        required=False,
        help="Manually select device: \'cuda:0\' or \'cpu\'"
    )
    parser.add_argument(
        "--quantize",
        default=None,
        type=str,
        required=False,
        help="If set: Do quantize dynamically model weights from float32 to qint8"
    )

    nrows = 3
    args = parser.parse_args()
    if args.device is not None:
        global device 
        device = torch.device(args.device)
        print('Device manually changed to: %s' %device)

    """
    df = pd.read_csv(args.data, nrows=nrows)
    df.to_csv('D:\\Documents\\recsys\\recsys2020\\tok_cls\\original.csv', index = False)
    return 0 
    df = pd.read_csv('D:\\Documents\\recsys\\recsys2020\\tok_cls\\tok_cls.csv')
    df.to_pickle('D:\\Documents\\recsys\\recsys2020\\tok_cls\\tok_cls_reduced.bz2')
    return 0
    """
    """
    print('done')
    input()
    df2= pd.read_csv('D:\\Documents\\recsys\\recsys2020\\tok_cls\\tok_cls.csv')
    for i in range(1000):
        df = df.append(df2)
    df.to_pickle('D:\\Documents\\recsys\\recsys2020\\tok_cls\\tok_cls.bz2')
    print('done')
    input()
    df = pd.read_csv(args.data, nrows=nrows)
    df.to_csv('D:\\Documents\\recsys\\recsys2020\\tok_cls\\tok_cls_ex.csv', index=False)
    input()
    print(len(df.index))
    df.to_parquet('D:\\Documents\\recsys\\recsys2020\\tok_cls\\tok_cls.parquet')
    df.to_pickle('D:\\Documents\\recsys\\recsys2020\\tok_cls\\tok_cls.pkl')
    compressions = ['gzip', 'bz2', 'zip', 'xz']
    for comp in compressions:
        df.to_pickle('D:\\Documents\\recsys\\recsys2020\\tok_cls\\tok_cls_'+str(comp)+'.pkl', compression=comp)
    df.to_pickle('D:\\Documents\\recsys\\recsys2020\\tok_cls\\tok_cls.gzip')
    df.to_pickle('D:\\Documents\\recsys\\recsys2020\\tok_cls\\tok_cls.bz2')
    df.to_pickle('D:\\Documents\\recsys\\recsys2020\\tok_cls\\tok_cls.zip')
    df.to_pickle('D:\\Documents\\recsys\\recsys2020\\tok_cls\\tok_cls.xz')
    compressors = ['blosc', 'bzip2', 'zlib']
    for compressor in compressors:
        df.to_hdf('D:\\Documents\\recsys\\recsys2020\\tok_cls\\tok_cls_'+str(compressor)+'.h5','table',complevel=9, complib=compressor, mode='w')
    df2 = pd.read_parquet('D:\\Documents\\recsys\\recsys2020\\tok_cls\\tok_cls.parquet')
    print(len(df2.index))
    print(df2.head())
    input()
    """

    if args.unique is not None:
        print('Reading...')
        df = pd.read_csv(args.data, sep='\u0001', header=None, nrows=nrows, usecols=[0])
        df.columns = ['Text_tokens']
        number_of_rows = len(df.index)
        df.drop_duplicates(inplace=True)
        print('Before drop:', number_of_rows,'After drop:',len(df.index),'\n% unique Text_tokens:',len(df.index)/number_of_rows)
        dummy = RecSysUtility('')
        print('Cleaning...')
        df['Text_tokens'] = df['Text_tokens'].apply(lambda x: dummy.clean_col(x))
        print(df.head())
        to_write = args.data.split('.tsv')[0] + '_uniqueTT.csv'
        print('Writing to %s' %to_write)
        df.to_csv(to_write, index=False, header = True)
        print('Done writing.')
        return 0
    # Initializing a BERT model
    model = BERT(pretrained=TrainingConfig._pretrained_bert, n_labels=TrainingConfig._num_labels, dropout_prob = TrainingConfig._dropout_prob, freeze_bert = True)

    # load del nostro modello per la fase di test,  andrebbe caricato questo direttamente al posto del pretrained bert
    checkpoint = torch.load(os.path.join(TrainingConfig._checkpoint_path, 'bert_model_test.pth'))
    model.load_state_dict(checkpoint)
    model.eval()

    if args.quantize is not None:
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

    filename = os.path.splitext(os.path.basename(args.data))[0] #filename without extension
    if args.jobn is None:
        print('Reading..')
        data = pd.read_csv(args.data, nrows=nrows, usecols=[args.tokcolumn])
        output_file = '../tok_cls/'+filename+'_tok_cls.bz2'
        compute_cls(data, model, args, output_file=output_file)
    else:
        if nrows is None:
            nrows = 10000000 #10M
            
        output_file = '../tok_cls/'+filename+'_tok_cls_'+ str(args.jobn) + '.bz2'
        print('Reading.. %d up to %d row' %(args.jobn*nrows, args.jobn*nrows + nrows -1))
        data = pd.read_csv(args.data, nrows=nrows, skiprows=range(1, args.jobn*nrows), usecols=[args.tokcolumn])
        data.columns = [args.tokcolumn]
        print(data.head(),'___ len:', len(data.index))
        compute_cls(data, model, args, output_file=output_file)
    

if __name__ == "__main__":
    main()
