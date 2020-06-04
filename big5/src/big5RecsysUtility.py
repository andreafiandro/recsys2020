import configSimo as c
import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class Big5Recsys:

    def __init__(self):
        self.tokenizer = c.TOKENIZER
        self.bert_pretrained = c.PRETRAINED_PYTORCH_BERT

    def from_text_tokens_to_text(
            self, text_tokens, print_token=False, print_text=False):
        '''
            Example of input
            import simo_recsys20_utilities.py as sru
            # open csv with pandas and store it as dataframe
            filename = "training_chunk_0.csv"
            dataset = pd.read_csv(filename)
            # print(dataset.head()) # serve a vedere come sono strutturati
            # in text_tokens salviamo il dato
            # come ci viene fornito dalla challenge
            # ovvero la lista di bert token separati da | come stringa
            text_tokens = dataset.iloc[758, 0]
            # print("Text tokens: ", text_tokens)
            original_tweet = sru.from_token_to_text(text_tokens,
                                                    print_token=False)
        '''
        text_token_list = []
        # rimozione \n poi facciamo una lista di token string
        # infine trasformiamo le stringhe in interi (indici nel vocabolario)
        text_token_list = self.split_tokens(text_tokens)
        if(print_token is True):
            print("Text token list: ", text_token_list)
        # ora la funzione che converte la lista di token in una stringa
        # ovvero nel tweet testuale originale
        token_list = text_token_list
        # print(list(tokenizer.vocab.keys())[token_list])
        original_tweet = ""
        flag = 0
        for token in token_list:
            # print(list(tokenizer.vocab.keys())[token])
            text_token = list(self.tokenizer.vocab.keys())[token]
            # print(text_token)
            flag += 1  # check if first token
            # gestione della spaziatura e ricostruzione word splitting
            if flag > 1:
                if len(text_token) > 2:  # caso token da word splitting
                    if text_token[1] == "#":
                        text_token = text_token[2:]
                        original_tweet = original_tweet + text_token
                    else:  # caso tweet mentions and tweet hashtag
                        if original_tweet[-1] == "@" \
                           or original_tweet[-1] == "#":
                            original_tweet = original_tweet + text_token
                        else:
                            original_tweet = original_tweet + " " + text_token
                else:
                    if original_tweet[-1] == "@" or original_tweet[-1] == "#":
                        original_tweet = original_tweet + text_token
                    else:
                        original_tweet = original_tweet + " " + text_token
            else:
                original_tweet = original_tweet + text_token
        if(print_text is True):
            print(original_tweet)
        return original_tweet

    def from_text_tokens_to_sentence_embedding(self, text_tokens):
        '''
        This function receive as input the first parameter in the dataset of
        the recsys2020 challenge "Text token".
        - it splits into a list of tokens
        - it creates a single embeddings for the whole sentece to avoid padding
        - it outouts a 768 dimension numpy array
        '''
        indexed_tokens = []
        indexed_tokens = self.split_tokens(text_tokens)
        # Mark each token as belonging to sentence "1". -> classification task
        segments_ids = [1] * len(indexed_tokens)
        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        # Load pre-trained model (weights)
        model = BertModel.from_pretrained(self.bert_pretrained)
        # Put the model in "evaluation" mode, meaning feed-forward operation.
        model.eval()
        # Predict hidden states features for each layer
        with torch.no_grad():
            encoded_layers, _ = model(tokens_tensor, segments_tensors)
        # Concatenate the tensors for all layers. We use `stack` here to
        # create a new dimension in the tensor.
        token_embeddings = torch.stack(encoded_layers, dim=0)
        # Remove dimension 1, the "batches".
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        # Swap dimensions 0 and 1.
        token_embeddings = token_embeddings.permute(1, 0, 2)
        # !!!!! AVG and NOT CLS!!!!
        # `encoded_layers` has shape [12 x 1 x 22 x 768]
        # `token_vecs` is a tensor with shape [22 x 768]
        token_vecs = encoded_layers[11][0]
        # Calculate the average of all 22 token vectors.
        sentence_embedding = torch.mean(token_vecs, dim=0)
        return sentence_embedding.numpy()

    def split_tokens(self, text_tokens):
        indexed_tokens = []
        text_tokens = text_tokens.rstrip("\n")
        indexed_tokens = text_tokens.split("|")
        indexed_tokens = list(map(int, indexed_tokens))
        return indexed_tokens

    def map_output_features(filename):
        '''
            Receive training_chunk_n.csv
            Map output features to 1 and 0 (NaN and timestamp)
            return dataframe mapped with text tokens and output
        '''
        # filename = "training_chunk_0.csv"
        dataset = pd.read_csv(filename)
        # print(dataset.shape)
        token_and_labels = \
            dataset.loc[:, ['Text_tokens',
                            'Reply_engagement_timestamp',
                            'Retweet_engagement_timestamp',
                            'Retweet_with_comment_engagement_timestamp',
                            'Like_engagement_timestamp']]
        # print(len(token_and_labels.index))
        for col in ['Reply_engagement_timestamp',
                    'Retweet_engagement_timestamp',
                    'Retweet_with_comment_engagement_timestamp',
                    'Like_engagement_timestamp']:
            token_and_labels[col] = \
                token_and_labels[col].\
                apply(lambda x: 1 if not pd.isnull(x) else 0)
        return token_and_labels

    def clean_and_pad_192(self, text_tokens):
        '''
        From text_tokens to a splitted and padded list of 192
        elements zero padded and without cls and sep bert token id
        '''
        indexed_tokens = []
        indexed_tokens = self.split_tokens(text_tokens)
        # strip cls 101 and sep 102
        cleaned = []
        for token in indexed_tokens:
            if(token == 101 or token == 102):
                continue
            else:
                cleaned.append(token)
        padded = []
        for i in range(c.PAD_MAX_LEN):
            if(i < len(cleaned)):
                padded.append(cleaned[i])
            else:
                padded.append(0)
        return padded

    def compute_big5(self, bc, text_tokens):
        '''
            before calling this funtion, in main you have to write
            these lines of code
            big5_scores = []
            server = BertServer(c.ARGS)
            server.start()
            bc = BertClient(ip='0.0.0.0')
            bu = Big5Recsys()
            bu.load_models()
            big5_scores = bu.compute_big5(bc, example_tokens)
        '''
        big5_scores = []
        tweet = self.from_text_tokens_to_text(text_tokens)
        tweetEmbeddings = bc.encode([tweet])
        result = self.model_O(torch.from_numpy(tweetEmbeddings[0]))
        big5_scores.append(round(result.item(), 3))
        result = self.model_C(torch.from_numpy(tweetEmbeddings[0]))
        big5_scores.append(round(result.item(), 3))
        result = self.model_E(torch.from_numpy(tweetEmbeddings[0]))
        big5_scores.append(round(result.item(), 3))
        result = self.model_A(torch.from_numpy(tweetEmbeddings[0]))
        big5_scores.append(round(result.item(), 3))
        result = self.model_N(torch.from_numpy(tweetEmbeddings[0]))
        big5_scores.append(round(result.item(), 3))
        return big5_scores

    def load_big5_model(self, trait):
        print("model_"+trait+" loading")
        model = nn.Sequential(nn.Linear(768, 300),
                              nn.ReLU(), nn.Linear(300, 1))
        model.load_state_dict(torch.load(c.PATH_TO_MODEL+"SentPers_"+trait))
        model.eval()
        return model

    def load_models(self):
        self.model_O = self.load_big5_model("O")
        self.model_C = self.load_big5_model("C")
        self.model_E = self.load_big5_model("E")
        self.model_A = self.load_big5_model("A")
        self.model_N = self.load_big5_model("N")

    def from_text_to_token_id(self, text):
        '''
        # how to use this function
        bu = Big5Recsys()
        text = "retweet"
        bu.from_text_to_token_id(text)
        '''
        # text = "Here is the sentence I want embeddings for."
        marked_text = "[CLS] " + text + " [SEP]"

        # Tokenize our sentence with the BERT tokenizer.
        tokenized_text = self.tokenizer.tokenize(marked_text)

        # Print out the tokens.
        print(tokenized_text)
        # Map the token strings to their vocabulary indeces.
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        # Display the words with their indeces.
        for tup in zip(tokenized_text, indexed_tokens):
            print('{:<12} {:>6,}'.format(tup[0], tup[1]))

    def create_topic_label_file_from_be(self, dfin, fileout, n_topics=20):
        # emb = pd.read_csv(filein, header=None)  # csv containing bert embeddings
        model = AgglomerativeClustering(n_clusters=n_topics)
        yhat = model.fit_predict(dfin)
        fo = open(fileout, "w")
        for y in yhat:
            fo.write(str(y)+"\n")
        fo.close()

    def write_emb_to_file(
            self, bc,
            filein, column_in_name, already_text,
            fileout, start_line
            ):
        be = open(fileout, "a")
        df = pd.read_csv(filein)
        to_emb = df.loc[:, column_in_name]
        for i in range(start_line, df.shape[0]):
            if already_text is False:  # bert_tokens | separated
                tweet = self.from_text_tokens_to_text(to_emb[i])
            else:
                tweet = to_emb[i]
            if pd.isnull(to_emb[i]):
                print("\n\n"+str(i)+"<----------------------------------- \n")
                continue
            tweetEmbeddings = bc.encode([tweet])
            emb = tweetEmbeddings[0]
            for elem in emb[:-1]:
                be.write(str(elem)+",")
            be.write(str(emb[-1])+"\n")
        be.close()


class MyClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.epochs = 100
        self.batch_size = 100
        self.loss_fn = nn.CrossEntropyLoss()
        self.model = nn.Sequential(
            nn.Linear(768, 300),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(300, int(output_size)))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def load_data(self, dfin, csv_target):
        #x = pd.read_csv(csv_input, header=None)
        dfsent = pd.read_csv(csv_target)
        # y = dfsent["sentiment"].astype("category").cat.codes
        y = dfsent["sentiment"].astype("category").cat.codes
        print(y.head)
        print(y.head)
        #inputs = torch.from_numpy(np.array(x))
        inputs = dfin.type(torch.FloatTensor)
        target = torch.from_numpy(np.array(y))
        target = target.type(torch.LongTensor)
        train_ds = TensorDataset(inputs, target)
        self.train_dl = DataLoader(train_ds, batch_size=self.batch_size,
                                   shuffle=True)

    def fit_and_save(self, model_fileout):
        for epoch in range(self.epochs):
            for xb, yb in self.train_dl:
                #print("xb", xb)
                pred = self.model(xb)
                #print("pred", pred)
                #print("yb", yb)
                loss = self.loss_fn(pred, yb)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            if(epoch+1) % 10 == 0:
                print('Epoch [%d/%d], Loss: %.4f' % (epoch+1, self.epochs,
                                                     loss.item()))
        torch.save(self.model.state_dict(), model_fileout)


class MergeClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        print("init")
        self.epochs = 10
        self.batch_size = 10
        self.loss_fn = nn.BCELoss()
        self.fc = nn.Linear(768, 300)
        self.fc1 = nn.Linear(300, 30)
        self.fc2 = nn.Linear(30+7, 1)
        self.out_act = nn.Sigmoid()

    def load_data(self, model, dfTens1, dfTens2, csv_target):
        print("load")
        self.optimizer = torch.optim.Adam(model.parameters(),  lr=0.001)
        #x = pd.read_csv(csv_input, header=None)
        dftl = Big5Recsys.map_output_features(csv_target)
        # y = dfsent["sentiment"].astype("category").cat.codes
        y = dftl["Like_engagement_timestamp"]
        y = y.iloc[0:100]
        inputs = dfTens1  # torch.from_numpy(np.array(x))
        inputs = inputs.type(torch.FloatTensor)
        inputs2 = dfTens2  # torch.from_numpy(np.array(x2))
        inputs2 = inputs2.type(torch.FloatTensor)
        target = torch.from_numpy(np.array(y))
        target = target.type(torch.FloatTensor)
        #print("inputs", inputs)
        #print("inputs2", inputs2)
        train_ds = TensorDataset(inputs, inputs2, target)
        self.train_dl = DataLoader(train_ds, batch_size=self.batch_size)

    def forward(self, emb, sentopbig):
        x = self.fc(emb)
        x = self.fc1(x)
        x = torch.cat((x, sentopbig), dim=1)
        x = self.fc2(x)
        y = self.out_act(x)
        return y

    def fit_and_save(self, model_fileout):
        for epoch in range(self.epochs):
            for batch_idx, (xb0, xb1, yb) in enumerate(self.train_dl):
                #print("xb0", xb0)
                #print("xb1", xb1)
                #print("yb", yb)
                #exit()
                pred = self.forward(xb0, xb1)
                print("pred", pred)
                print("yb", yb)
                # yb = yb.view(yb.size()[0], 1)
                #exit()
                loss = self.loss_fn(pred.type(torch.FloatTensor), yb.type(torch.FloatTensor))
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            if(epoch+1) % 1 == 0:
                print('Epoch [%d/%d], Loss: %.4f' % (epoch+1, self.epochs,
                                                     loss.item()))
        torch.save(self.model.state_dict(), model_fileout)


class binaryClassification(nn.Module):
    def __init__(self):
        super(binaryClassification, self).__init__()

        self.layer_1 = nn.Linear(768, 300)
        self.layer_2 = nn.Linear(300, 30)
        self.layer_out = nn.Linear(30+7, 1)

        self.lrelu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(300)
        self.batchnorm2 = nn.BatchNorm1d(30)

    def forward(self, inputs, inputs2):
        x = self.lrelu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.lrelu(self.layer_2(x))
        