import pandas as pd
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased',
                                          do_lower_case=False)


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
    text_tokens = text_tokens.rstrip("\n")
    text_token_list = text_tokens.split("|")
    text_token_list = list(map(int, text_token_list))
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
