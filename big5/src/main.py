import config as c
# from big5RecsysUtility import Big5Recsys
from bert_serving.client import BertClient
from bert_serving.server import BertServer
import pandas as pd
import os
# from recsysUtility import RecSysUtility
'''
    Run bert-as-a-service server/client to compute BERT embeddings.
    Load the SentencePersonality model to compute
    big5 personality traits from the sentence embedding.
'''
'''
# Start server/client bert-as-a-service.
# Look at config.py to check/modify parameters
server = BertServer(c.ARGS)
server.start()
bc = BertClient(ip=c.BC_IP)

# Read dataset of recsys2020 challenge as stored by A.Fiandro
# input/training_chunk_0.csv
# Select the first column of the dataset Text_tokens
# 101|1234|56789|102|987|32|102 is an example string of Text_tokens
# 101 = ['CLS'] 102 = ['SEP'] look at BERT for more info on WordPiece
df = pd.read_csv(c.FILENAME)
token_col = df.loc['Text tokens']

big5_scores = []
fout = open("big5_chunk_"+c.CHUNK_NUM+".csv", "a")

for i in range(c.START_LINE, token_col.shape[0]):
    print("progress ", str(i), "/", str(token_col.shape[0]))
    tokens_string = df.loc[i, 'text']
    big5_scores = sru.compute_big5(bc, tokens_string)
    for j in range(4):
        fout.write(str(big5_scores[j]))
        fout.write(",")
    fout.write(str(big5_scores[4]))
    fout.write("\n")

fout.close()

bc.close()
cmd = 'bert-serving-terminate -port ' + c.PORT_OUT
os.system(cmd)


# in text_tokens salviamo il dato come ci viene fornito dalla challenge
# ovvero la lista di bert token separati da | come stringa
# padded = sru.clean_and_pad_192(example_tokens)
# token_and_labels = sru.map_output_features(filename)
# print(token_and_labels.shape[0])
# df = token_and_labels.loc[:, ['Text tokens', 'Reply engagement timestamp']]
# df.columns = ['text', 'label']
# example_tokens = df.loc[200, 'text']
'''

# rcUtils = RecSysUtility('../input/val.tsv')
# rcUtils.create_chunk_csv()
df = pd.read_csv("../input/val/val_chunk_0.csv")
print(df.head())

text_tokens = df.loc[:, 'Text_tokens']
print(text_tokens.head())

bu = Big5Recsys()

big5_scores = []
server = BertServer(c.ARGS)
server.start()
bc = BertClient(ip='0.0.0.0')
bu = Big5Recsys()
bu.load_models()
big5_scores = bu.compute_big5(bc, text_tokens[0])
print(big5_scores)

bc.close()
myCmd = 'bert-serving-terminate -port 5555'
os.system(myCmd)
