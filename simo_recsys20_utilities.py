import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertTokenizer, BertModel
import pandas as pd

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased',
                                          do_lower_case=False)
PAD_MAX_LEN = 192
PATH_TO_MODEL = "../BERT/big5/Models/"


def from_token_to_text(text_tokens, print_token=True):
    '''
        Example of input
        import simo_recsys20_utilities.py as sru
        # open csv with pandas and store it as dataframe
        filename = "training_chunk_0.csv"
        dataset = pd.read_csv(filename)
        # print(dataset.head()) # serve a vedere come sono strutturati
        # in text_tokens salviamo il dato come ci viene fornito dalla challenge
        # ovvero la lista di bert token separati da | come stringa
        text_tokens = dataset.iloc[758, 0]
        # print("Text tokens: ", text_tokens)
        original_tweet = sru.from_token_to_text(text_tokens, print_token=False)
    '''
    text_token_list = []
    # rimozione \n poi facciamo una lista di token string
    # infine trasformiamo le stringhe in interi (indici nel vocabolario)
    text_token_list = split_tokens(text_tokens)
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
        text_token = list(tokenizer.vocab.keys())[token]
        # print(text_token)
        flag += 1  # check if first token
        # gestione della spaziatura e ricostruzione word splitting
        if flag > 1:
            if len(text_token) > 2:  # caso token da word splitting
                if text_token[1] == "#":
                    text_token = text_token[2:]
                    original_tweet = original_tweet + text_token
                else:  # caso tweet mentions and tweet hashtag
                    if original_tweet[-1] == "@" or original_tweet[-1] == "#":
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
    print(original_tweet)
    return original_tweet


def from_text_to_sentence_embedding(text_tokens):
    '''
        This function receive as input the first parameter in the dataset of
        the recsys2020 challenge "Text token".
        - it splits into a list of tokens
        - it creates a single embeddings for the whole sentece to avoid padding
        - it outouts a 768 dimension numpy array
    '''
    indexed_tokens = []
    indexed_tokens = split_tokens(text_tokens)
    # Mark each token as belonging to sentence "1". -> classification task
    segments_ids = [1] * len(indexed_tokens)
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    # Load pre-trained model (weights)
    model = BertModel.from_pretrained('bert-base-multilingual-cased')

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


def split_tokens(text_tokens):
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
        dataset.loc[:, ['Text tokens',
                        'Reply engagement timestamp',
                        'Retweet engagement timestamp',
                        'Retweet with comment engagement timestamp',
                        'Like engagement timestamp']]
    # print(len(token_and_labels.index))
    for col in ['Reply engagement timestamp', 'Retweet engagement timestamp',
                'Retweet with comment engagement timestamp',
                'Like engagement timestamp']:
        token_and_labels[col] = token_and_labels[col].apply(lambda x: 1 if not
                                                            pd.isnull(x)
                                                            else 0)
    return token_and_labels


# seconda fase bert e cls "simple transformer libreria"
def clean_and_pad_192(text_tokens):
    indexed_tokens = []
    indexed_tokens = split_tokens(text_tokens)
    # strip cls 101 and sep 102
    cleaned = []
    for token in indexed_tokens:
        if(token == 101 or token == 102):
            continue
        else:
            cleaned.append(token)
    padded = []
    for i in range(PAD_MAX_LEN):
        if(i < len(cleaned)):
            padded.append(cleaned[i])
        else:
            padded.append(0)
    return padded


def compute_big5(bc, text_tokens):
    '''
        before calling this funtion, in main you have to write
        these lines of code
        big5_scores = []
        args = get_args_parser().parse_args(['-model_dir',
                '../dataset/multi_cased_L-12_H-768_A-12',
                '-port', '5555', '-port_out', '5556','-max_seq_len','NONE',
                '-mask_cls_sep','-num_worker=1',  #removed -cpu -> runs on gpu
                '-device_map=0', '-pooling_strategy', 'CLS_TOKEN'])
        server = BertServer(args)
        server.start()
        bc = BertClient(ip='0.0.0.0')

        big5_scores = sru.compute_big5(bc, example_tokens)
    '''
    big5_scores = []
    tweet = from_token_to_text(text_tokens)
    tweetEmbeddings = bc.encode([tweet])
    result = model_O(torch.from_numpy(tweetEmbeddings[0]))
    big5_scores.append(round(result.item(), 3))
    result = model_C(torch.from_numpy(tweetEmbeddings[0]))
    big5_scores.append(round(result.item(), 3))
    result = model_E(torch.from_numpy(tweetEmbeddings[0]))
    big5_scores.append(round(result.item(), 3))
    result = model_A(torch.from_numpy(tweetEmbeddings[0]))
    big5_scores.append(round(result.item(), 3))
    result = model_N(torch.from_numpy(tweetEmbeddings[0]))
    big5_scores.append(round(result.item(), 3))
    return big5_scores


def load_big5_model(trait):
    print("model loading")
    model = nn.Sequential(nn.Linear(768, 300),
                          nn.ReLU(), nn.Linear(300, 1))
    model.load_state_dict(torch.load(PATH_TO_MODEL+"SentPers_"+trait))
    model.eval()
    return model


model_O = load_big5_model("O")
model_C = load_big5_model("C")
model_E = load_big5_model("E")
model_A = load_big5_model("A")
model_N = load_big5_model("N")
