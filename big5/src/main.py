import configSimo as c
from big5RecsysUtility import Big5Recsys, MyClassifier, MergeClassifier
from bert_serving.client import BertClient
from bert_serving.server import BertServer
import pandas as pd
import os
from sklearn.mixture import GaussianMixture
from numpy import unique, where
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering
# from recsysUtility import RecSysUtility
import torch
import torch.nn as nn
import numpy as np
from bert_test import cls_isa  # bertisa

bu = Big5Recsys()
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


""" df = pd.read_csv("../input/training_chunk_140.csv")
seriesObj = df.apply(lambda x: False if str(x['Hashtags']) == "0"
                     else True, axis=1)
numOfRows = len(seriesObj[seriesObj == True].index)
print('Number of Rows in dataframe in which hashtags exist: ', numOfRows) """

""" counter = 0
for i in range(df.shape[0]):
    if str(df.loc[i, ['Hashtags']].item()) != "0":
        counter = counter+1

print(counter) """


'''
'Text_tokens', 'Hashtags', 'Tweet_id', 'Present_media', 'Present_links',
       'Present_domains', 'Tweet_type', 'Language', 'Timestamp', 'User_id',
       'Follower_count', 'Following_count', 'Is_verified',
       'Account_creation_time', 'User_id_engaging', 'Follower_count_engaging',
       'Following_count_engaging', 'Is_verified_engaging',
       'Account_creation_time_engaging', 'Engagee_follows_engager',
       'Reply_engagement_timestamp', 'Retweet_engagement_timestamp',
       'Retweet_with_comment_engage'
'''

# topic extraxtion from twitter
'''
print(yhat)
# retrieve unique clusters
clusters = unique(yhat)
print(len(clusters))

df = pd.read_csv("../input/training_chunk_140.csv")
text_tokens = df.loc[:, 'Text_tokens']
bu = Big5Recsys()

for cluster in clusters:
    row_ix = where(yhat == cluster)
    print(row_ix[0])
    for i in row_ix[0]:
        tweet = bu.from_text_tokens_to_text(text_tokens[i])
        print(i, tweet)
'''
# create fit and save model for sentiment ###
# dfinTens = cls_isa()
# mc = MyClassifier(768, 3)
# mc.load_data(dfinTens, "sent_text_tokens.csv")
# mc.fit_and_save("../models/sentiment")
# print("ended")

# load and predict sentiment ###
'''
model = nn.Sequential(
            nn.Linear(768, 300),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(300, 3))
model.load_state_dict(torch.load("../models/sentiment"))
model.eval()
df = pd.read_csv("../output/sentiment/emb.csv")
example = df.iloc[2, :]
#print(example.to_numpy())
result = torch.from_numpy(example.to_numpy())
result = result.type(torch.FloatTensor)
result = model(result)
print(np.argmax(result.detach().numpy()))
'''
# agglomerative clustering to create topic labels #

# filein = "../output/emb/emb_140.csv"
# dfinTens = cls_isa()
# dfin = dfinTens.numpy()
# fileout = "../output/topic/y_139.csv"
# bu.create_topic_label_file_from_be(dfin, fileout, 15)
# din = pd.read_csv(filein, header=None)
# dout = pd.read_csv(fileout, header=None)
# print(din.shape, dout.shape)

# create fit and save model for topic ###
# mc = MyClassifier(768, 15)
# mc.load_data(dfinTens, "../output/topic/y_139.csv")
# mc.fit_and_save("../models/topic")
# print("ended")


modelsent = nn.Sequential(
            nn.Linear(768, 300),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(300, 3))
modelsent.load_state_dict(torch.load("../models/sentiment"))
modelsent.eval()

modeltopic = nn.Sequential(
            nn.Linear(768, 300),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(300, 15))
modeltopic.load_state_dict(torch.load("../models/topic"))
modeltopic.eval()

model_O = bu.load_big5_model("O")
model_C = bu.load_big5_model("C")
model_E = bu.load_big5_model("E")
model_A = bu.load_big5_model("A")
model_N = bu.load_big5_model("N")

dfTens = cls_isa()
o = model_O(dfTens)
con = model_C(dfTens)
e = model_E(dfTens)
a = model_A(dfTens)
n = model_N(dfTens)
t = torch.argmax(modeltopic(dfTens), dim=1)
t = t.view(t.size()[0], 1)
s = torch.argmax(modelsent(dfTens), dim=1)
s = s.view(s.size()[0], 1)
catT = torch.cat((o, con, e, a, n, t.type(torch.FloatTensor), s.type(torch.FloatTensor)), dim=1)
""" print("O: ", o)
print("Topic: ", t)
print("Sentiment: ", s)
print(" cat ", catT)
 """
'''
mecl = MergeClassifier()
mecl.load_data(mecl, dfTens, catT, "../input/training_chunks/training_chunk_134.csv")
mecl.fit_and_save("models/like")
print("ended")
'''
result = catT.detach().numpy()
#print(catT.detach().numpy())

res = np.asarray(result)
np.savetxt("../output/oceants/val_0p1.csv", res, delimiter=",", fmt='%5.5f')