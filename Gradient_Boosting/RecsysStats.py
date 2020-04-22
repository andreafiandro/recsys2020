import dask.dataframe as dd
import gc
import logging
from dask.diagnostics import ProgressBar
import pandas as pd

class RecSysStats:

    def __init__(self, training_file):
        logging.basicConfig(filename='statistics.log',level=logging.INFO)
        ProgressBar().register()
        self.training_file = training_file
        self.col_names_val = ['Text_tokens', 'Hashtags', 'Tweet_id', 'Present_media', 'Present_links', 'Present_domains', 'Tweet_type', 'Language', 'Timestamp',
        'User_id', 'Follower_count', 'Following_count', 'Is_verified', 'Account_creation_time',
        'User_id_engaging', 'Follower_count_engaging', 'Following_count_engaging', 'Is_verified_engaging', 'Account_creation_time_engaging',
        'Engagee_follows_engager']

        self.col_names_training = ['Text_tokens', 'Hashtags', 'Tweet_id', 'Present_media', 'Present_links', 'Present_domains', 'Tweet_type', 'Language', 'Timestamp',
        'User_id', 'Follower_count', 'Following_count', 'Is_verified', 'Account_creation_time',
        'User_id_engaging', 'Follower_count_engaging', 'Following_count_engaging', 'Is_verified_engaging', 'Account_creation_time_engaging',
        'Engagee_follows_engager', 'Reply_engagement_timestamp', 'Retweet_engagement_timestamp', 'Retweet_with_comment_engagement_timestamp', 'Like_engagement_timestamp']
        
    """
    ------------------------------------------------------------------------------------------
    DATASET STATISTICS
    ------------------------------------------------------------------------------------------
    """

    def count_n_lines(self):
        dd_input = dd.read_csv(self.training_file, sep='\u0001', header=None)
        n_rows = dd_input.shape[0].compute()
        self.print_and_log('The Dataframe has {} lines'.format(n_rows))
        return n_rows

    def count_n_tweets(self, isVal=False):
        dd_input = dd.read_csv(self.training_file, sep='\u0001', header=None)
        if(isVal):
            dd_input.columns = self.col_names_val
        else:
            dd_input.columns = self.col_names_training
        n_tweets = dd_input['Tweet_id'].unique().compute()
        self.print_and_log('The Dataframe has {} tweets'.format(n_tweets.shape[0]))
        return
    
    def get_all_authors(self, isVal=False):
        print('I get all the tweet authors')
        dd_input = dd.read_csv(self.training_file, sep='\u0001', header=None)
        dd_input = self.process_chunk_tsv(dd_input, isVal=isVal)
        list_authors = dd_input['User_id'].unique().compute()
        self.print_and_log('The Dataframe has {} authors'.format(list_authors.shape[0]))
        return set(list_authors)
    
    def get_all_users(self, isVal=False):
        print('I get all the tweet user')
        dd_input = dd.read_csv(self.training_file, sep='\u0001', header=None)
        dd_input = self.process_chunk_tsv(dd_input, isVal=isVal)
        list_authors = dd_input['User_id_engaging'].unique().compute()
        self.print_and_log('The Dataframe has {} users'.format(list_authors.shape[0]))
        return set(list_authors)

    def user_or_author(self, isVal=False):
        list_authors = self.get_all_authors(isVal)
        list_users = self.get_all_users(isVal)
        
        print('Compute intersection')
        user_and_author = list_authors.intersection(list_users)
        self.print_and_log('{} are both user and authors'.format(len(user_and_author)))
        del user_and_author
        gc.collect()

        print('Compute only user')
        only_user = list_users.difference(list_authors)
        self.print_and_log('{} are only users'.format(len(only_user)))
        del only_user
        gc.collect()
        
        print('Compute only authors')
        only_authors = list_authors.difference(list_users)
        self.print_and_log('{} are only authors'.format(len(only_authors)))
        del only_authors
        gc.collect()
        return
    
    def count_action_type(self, label, isVal=False):
        print('I get all the {} actions'.format(label))
        label_pandas = label + '_engagement_timestamp'
        dd_input = dd.read_csv(self.training_file, sep='\u0001', header=None)
        dd_input = self.process_chunk_tsv(dd_input, isVal=isVal)
        df_not_null = dd_input[label_pandas][~dd_input[label_pandas].isna()].compute()
        self.print_and_log('There are {} action of type {}'.format(df_not_null.shape[0], label))
        return

    
    def get_validation_users(self, val_file):
        print('I get all the users from validation')
        dd_input = dd.read_csv(val_file, sep='\u0001', header=None)
        dd_input = self.process_chunk_tsv(dd_input, isVal=True)
        list_users = dd_input['User_id_engaging'].unique().compute()
        self.print_and_log('The Validation Set has {} users'.format(list_users.shape[0]))
        return set(list_users)


    def train_or_val(self, val_file):
        list_training = self.get_all_users()
        list_validation = self.get_validation_users(val_file)

        print('Compute intersection')
        training_and_val = list_training.intersection(list_validation)
        self.print_and_log('{} are both in training and validation'.format(len(training_and_val)))
        del training_and_val
        gc.collect()

        print('Compute only training')
        only_training = list_training.difference(list_validation)
        self.print_and_log('{} are only in the training'.format(len(only_training)))
        del only_training
        gc.collect()
        
        print('Compute only validation')
        only_validation = list_validation.difference(list_training)
        self.print_and_log('{} are only in the validation'.format(len(only_validation)))
        del only_validation
        gc.collect()
        return

    # Da risolvere dipendenze circolari
    def print_and_log(self, to_print):
        logging.info(to_print)
        print(to_print)
        return

    def clean_col(self, x):
        """
            Used to clean the columns separated by \t to make them readable
        """
        if(pd.isna(x)):
            return x
        else:
            return x.replace('\t', '|')  

    def process_chunk_tsv(self, df, col_to_clean=['Text_tokens', 'Hashtags', 'Present_media', 'Present_links', 'Present_domains'], isVal=False):
        if(isVal):
            df.columns = self.col_names_val
        else:
            df.columns = self.col_names_training

        # Convert boolean to 1 / 0
        df['Is_verified'] = df['Is_verified'].apply(lambda  x: 1 if x else 0)
        df['Is_verified_engaging'] = df['Is_verified_engaging'].apply(lambda  x: 1 if x else 0)
        
        for col in col_to_clean:
            df[col] = df[col].apply(lambda x: self.clean_col(x))
            df[col] = df[col].fillna(0)
        return df

    def get_all_retweet(self, validation_file):
        dd_val = dd.read_csv(validation_file, sep='\u0001', header=None)
        dd_val = self.process_chunk_tsv(dd_val, isVal=True)
        # Tutti gli autori che hanno condiviso un tweet
        print('Cerco gli utenti con il doppio ruolo Autore/User')
        user_engaging = dd_val['User_id_engaging'].compute()
        dd_authors = dd_val[dd_val['User_id'].isin(user_engaging)].compute()
        print('Filtro quelli che hanno fatto almeno un retweet')
        dd_authors = dd_authors[dd_authors['Tweet_type'] == 'Retweet']
        print('Tengo solo Id + Tweet con cui hanno interagito')
        dd_authors = dd_authors[['User_id', 'Text_tokens']]
        print(dd_authors.head())
        print('Merge tra le azioni del validation da prevedere e autori che hanno un retweet in futuro')
        dd_compare = dd_val.merge(dd_authors, left_on=['User_id_engaging'], right_on=['User_id'], suffixes=('_attuale', '_prec')).compute()
        dd_compare.to_csv('tmp.csv')
        #print('Tengo solo quelli in cui i token corrispondono')
        print('Prevedo retweet di retweet')
        df_retweet = dd_compare[dd_compare['Text_tokens_attuale'] == dd_compare['Text_tokens_prec']]
        print('Prevedo retweet di tweet Top Level')
        dd_compare['Text_tokens_attuale'] = dd_compare['Text_tokens_attuale'].apply(lambda x: x.replace('101|', ''))
        dd_compare['Text_tokens_prec'] = dd_compare['Text_tokens_prec'].apply(lambda x: x.replace('101|56898|', ''))
        df_retweet_top = dd_compare[dd_compare['Text_tokens_attuale'] == dd_compare['Text_tokens_prec']]

        print('Ho trovato {} retweet sicuri'.format(df_retweet.shape[0]))
        df_retweet.to_csv('retweet_100.csv')

        print('Ho trovato {} retweet toplevel sicuri'.format(df_retweet_top.shape[0]))
        df_retweet_top.to_csv('retweet_top_100.csv')

        return


