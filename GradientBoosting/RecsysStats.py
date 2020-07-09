import dask.dataframe as dd
import gc
import logging
from dask.diagnostics import ProgressBar
import pandas as pd
import json
from tqdm import tqdm
import os

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
    
    def get_ntweets_for_user(self, isVal=False):
        dd_input = dd.read_csv(self.training_file, sep='\u0001', header=None)
        dd_input = self.process_chunk_tsv(dd_input, isVal=isVal)
        dd_input = dd_input[['User_id_engaging', 'Tweet_id']]
        df_result = dd_input.groupby('User_id_engaging')['Tweet_id'].count().reset_index().compute()
        self.print_and_log(df_result.head())
        df_result.to_csv('UserAndTweets.csv', index=False)
        #self.print_and_log('Mean #Tweets {}'.format(df_result['Tweet_id'].meas))

        return

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

    def count_pseudo_negative(self):
        df = dd.read_csv(self.training_file, sep='\u0001', header=None)
        df = self.process_chunk_tsv(df)
        df = df [['Reply_engagement_timestamp', 'Retweet_engagement_timestamp', 'Retweet_with_comment_engagement_timestamp', 'Like_engagement_timestamp']]
        df = df[df['Reply_engagement_timestamp'].isna() & df['Retweet_engagement_timestamp'].isna() & df['Retweet_with_comment_engagement_timestamp'].isna() & df['Like_engagement_timestamp'].isna()].compute()
        self.print_and_log('There are {} pseudo negative'.format(df.shape[0]))
        return

    
    def get_validation_users(self, val_file):
        print('I get all the users from validation')
        dd_input = dd.read_csv(val_file, sep='\u0001', header=None)
        dd_input = self.process_chunk_tsv(dd_input, isVal=True)
        list_users = dd_input['User_id_engaging'].unique().compute()
        self.print_and_log('The Validation Set has {} users'.format(list_users.shape[0]))
        return set(list_users)

    def get_validation_author(self, val_file):
        print('I get all the authors from validation')
        dd_input = dd.read_csv(val_file, sep='\u0001', header=None)
        dd_input = self.process_chunk_tsv(dd_input, isVal=True)
        list_users = dd_input['User_id'].unique().compute()
        self.print_and_log('The Validation Set has {} authors'.format(list_users.shape[0]))
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

    """
        Generazione degli encoding su tutto il dataset
    """

    def generate_user_encoding(self, validation_file):
        """
            Funzione per trasformare gli hash che rappresentano gli utenti in interi
            Output: scrive su disco il file user_encoding.json
        """
        print('Ottengo gli user per il training')
        training_user = self.get_all_users()
        #print('Salvo su file...')
        #f = open("training_user","w")
        #f.write(training_user)

        print('Ottengo gli user per il validation')
        validation_user = self.get_validation_users(validation_file)
        #print('Salvo su file...')
        #f = open("validation_user","w")
        #f.write(validation_user)

        print('Merge dei due gruppi di utenti')
        all_users = training_user.union(validation_user)
        print('Genero encoding...')
        user_dict = {}
        counter = 0 
        for i in all_users:
            user_dict[i] = counter
            counter += 1
        print('Scrivo user encoding su file...')
        f = open("user_encoding.json","w")
        f.write(json.dumps(user_dict))
        f.close()
        return

    def generate_author_encoding(self, validation_file):
        
        """
            Funzione per trasformare gli hash che rappresentano gli autori in interi
            Output: scrive su disco il file author_encoding.json
        """
        print('Ottengo gli authors per il training')
        training_user = self.get_all_authors()
        #print('Salvo su file...')
        #f = open("training_user","w")
        #f.write(training_user)

        print('Ottengo gli authors per il validation')
        validation_user = self.get_validation_author(validation_file)
        #print('Salvo su file...')
        #f = open("validation_user","w")
        #f.write(validation_user)

        print('Merge dei due gruppi di autori')
        all_authors = training_user.union(validation_user)
        print('Genero encoding...')
        authors_dict = {}
        counter = 0 
        for i in all_authors:
            authors_dict[i] = counter
            counter += 1
        print('Scrivo author encoding su file...')
        f = open("author_encoding.json","w")
        f.write(json.dumps(authors_dict))
        f.close()
        return

    def generate_hashtag_encoding(self, validation_file):
        """
            Funzione per generare un encoding di tutti gli hashtags su file
            Output: scrive su disco un file hashtags_encoding.json
        """
        print('I get all the hashtags from training')
        dd_input = dd.read_csv(self.training_file, sep='\u0001', header=None)
        dd_input = self.process_chunk_tsv(dd_input)
        print('Tolgo quelli nulli e li splitto')
        dd_input = dd_input[['Hashtags']]
        hashtags = dd_input[~dd_input['Hashtags'].isna()]['Hashtags'].compute()
        lista_hashtag = hashtags.to_list()
        print('Creo il set di hashtags unici')
        set_hashtag = set()
        for h in tqdm(lista_hashtag):
            if(isinstance(h, str)):
                line_ht = h.split('|')
                for a in line_ht:
                    set_hashtag.add(a)

        print('I get all the hashtags from validation')
        dd_input = dd.read_csv(validation_file, sep='\u0001', header=None)
        dd_input = self.process_chunk_tsv(dd_input, isVal=True)
        print('Tolgo quelli nulli e li splitto')
        dd_input = dd_input[['Hashtags']]
        hashtags = dd_input[~dd_input['Hashtags'].isna()]['Hashtags'].compute()
        
        print('Continuo il set di hashtags unici')
        for h in tqdm(lista_hashtag):
            if(isinstance(h, str)):
                line_ht = h.split('|')
                for a in line_ht:
                    set_hashtag.add(a)

        print('Genero encoding...')
        hashtags_dic = {}
        counter = 0
        for i in set_hashtag:
            hashtags_dic[i] = counter
            counter += 1
        print('Scrivo hashtags encoding su file...')
        f = open("hashtags_encoding.json","w")
        f.write(json.dumps(hashtags_dic))
        f.close()

        return set_hashtag

    def generate_language_encoding(self, validation_file):
        """
            Funzione per generare un encoding di tutti gli hashtags su file
            Output: scrive su disco un file language_encoding.json
        """
        print('I get all the language from training')
        dd_input = dd.read_csv(self.training_file, sep='\u0001', header=None)
        dd_input = self.process_chunk_tsv(dd_input)
        print('Tolgo quelli nulli e li splitto')
        dd_input = dd_input[['Language']]
        languages = dd_input[~dd_input['Language'].isna()]['Language'].compute()
        lista_languages = languages.to_list()
        print('Creo il set di Language unici')
        set_language = set()
        for h in tqdm(lista_languages):
            if(isinstance(h, str)):
                line_ht = h.split('|')
                for a in line_ht:
                    set_language.add(a)

        print('I get all the Language from validation')
        dd_input = dd.read_csv(validation_file, sep='\u0001', header=None)
        dd_input = self.process_chunk_tsv(dd_input, isVal=True)
        print('Tolgo quelli nulli e li splitto')
        dd_input = dd_input[['Language']]
        languages = dd_input[~dd_input['Language'].isna()]['Language'].compute()
        
        print('Continuo il set di Language unici')
        for h in tqdm(lista_languages):
            if(isinstance(h, str)):
                line_ht = h.split('|')
                for a in line_ht:
                    set_language.add(a)

        print('Genero encoding...')
        language_dic = {}
        counter = 0
        for i in set_language:
            language_dic[i] = counter
            counter += 1
        print('Scrivo Language encoding su file...')
        f = open("language_encoding.json","w")
        f.write(json.dumps(language_dic))
        f.close()

        return language_dic

    def generate_author_features(self, validation):
        """
            Creo una sparse matrix in cui ci sono tutti gli hashtag utilizzati da ogni author
        """
        print('Genero le features sul training set')
        dd_input = dd.read_csv(self.training_file, sep='\u0001', header=None)
        dd_input = self.process_chunk_tsv(dd_input)
        dd_input = dd_input[['User_id', 'Hashtags']]
        dd_input = dd_input.dropna(subset=['Hashtags'])
        dd_input = dd_input[dd_input['Hashtags'] != 0]
        dd_input['Hashtags'] = dd_input['Hashtags'].apply(lambda x: x.split('|') if(isinstance(x, str)) else [])
        dd_input = dd_input.explode('Hashtags')
        dd_input = dd_input.dropna(subset=['Hashtags'])
        dd_input = dd_input.drop_duplicates().compute()

        print('Genero le features sul validation set')
        dd_val = dd.read_csv(validation, sep='\u0001', header=None)
        dd_val = self.process_chunk_tsv(dd_val, isVal=True)
        dd_val = dd_val[['User_id', 'Hashtags']]
        dd_val = dd_val.dropna(subset=['Hashtags'])
        dd_val = dd_val[dd_val['Hashtags'] != 0]
        dd_val['Hashtags'] = dd_val['Hashtags'].apply(lambda x: x.split('|') if(isinstance(x, str)) else [])
        dd_val = dd_val.explode('Hashtags')
        dd_val = dd_val.dropna(subset=['Hashtags'])
        dd_val = dd_val.drop_duplicates().compute()


        print('Concateno i due df')
        dd_input = pd.concat([dd_input, dd_val], axis=0, ignore_index=True)

        dd_input = dd_input.drop_duplicates()
        print('Faccio encoding degli autori')
        
        json_author = open("author_encoding.json", "r")
        author_dic = json.load(json_author)
        dd_input['User_id'] = dd_input['User_id'].map(author_dic)

        print('Faccio encoding degli hashtags')
        json_hashtag = open("hashtag_encoding.json", "r")
        hashtag_dic = json.load(json_hashtag)
        dd_input['Hashtags'] = dd_input['Hashtags'].map(hashtag_dic)

        dd_input.to_csv('author_hashtag_mapping.csv', index=False, header = False)

        return
    
    def get_max_min_engagement_date(self):
        dd_input = dd.read_csv(self.training_file, sep='\u0001', header=None)
        dd_input = self.process_chunk_tsv(dd_input)
        dd_input = dd_input[['Reply_engagement_timestamp', 'Retweet_engagement_timestamp', 'Retweet_with_comment_engagement_timestamp', 'Like_engagement_timestamp']]
        for label in ['Reply', 'Retweet', 'Retweet_with_comment', 'Like']:
            col_label = label + '_engagement_timestamp'
            dd_label = dd_input[[col_label]]
            dd_label = dd_label[~dd_label[col_label].isna()]
            max_timestamp = dd_label[col_label].max().compute()
            min_timestamp = dd_label[col_label].min().compute()
            self.print_and_log('Label {} / Min Timestamp: {} / Max Timestamp: {}'.format(label, min_timestamp, max_timestamp))
        
        return


    """
        FEATURE ENGINEERING
    """

    def generate_user_features(self, validation, encode = False, grouped = True):
        """
            Creo una sparse matrix in cui ci sono tutte le lingue con cui ha interagito ogni user
        """
        print('Genero le features sul validation set')
        dd_val = dd.read_csv(validation, sep='\u0001', header=None)
        dd_val = self.process_chunk_tsv(dd_val, isVal=True)
        dd_val = dd_val[['User_id_engaging', 'Language']]
        dd_val = dd_val.dropna(subset=['Language'])
        dd_val = dd_val[dd_val['Language'] != 0]
        dd_val['Language'] = dd_val['Language'].apply(lambda x: x.split('|') if(isinstance(x, str)) else [])
        dd_val = dd_val.explode('Language')
        dd_val = dd_val.dropna(subset=['Language'])
        
        if(grouped):
            dd_val = dd_val.drop_duplicates().groupby('User_id_engaging')['Language'].apply(list).reset_index().compute()
        else:
            dd_val = dd_val.drop_duplicates().compute()

        print(dd_val.head())
        
        print('Genero le features sul training set')
        dd_input = dd.read_csv(self.training_file, sep='\u0001', header=None)
        dd_input = self.process_chunk_tsv(dd_input)
        dd_input = dd_input[['User_id_engaging', 'Language']]
        dd_input = dd_input.dropna(subset=['Language'])
        dd_input = dd_input[dd_input['Language'] != 0]
        dd_input['Language'] = dd_input['Language'].apply(lambda x: x.split('|') if(isinstance(x, str)) else [])
        dd_input = dd_input.explode('Language')
        dd_input = dd_input.dropna(subset=['Language'])
        if(grouped):
            dd_input = dd_input.drop_duplicates().groupby('User_id_engaging')['Language'].apply(list).reset_index().compute()
        else:
            dd_input = dd_input.drop_duplicates().compute()

    
        print('Concateno i due df')
        dd_input = pd.concat([dd_input, dd_val], axis=0, ignore_index=True)
        
        if(encode):
            dd_input = dd_input.drop_duplicates()
            print('Faccio encoding degli user')
            
            json_user = open("user_encoding.json", "r")
            user_dic = json.load(json_user)
            dd_input['User_id_engaging'] = dd_input['User_id_engaging'].map(user_dic)

            print('Faccio encoding delle lingue')
            json_language = open("language_encoding.json", "r")
            language_dic = json.load(json_language)
            dd_input['Language'] = dd_input['Language'].map(language_dic)
        dd_input.to_csv('user_language_mapping.csv', index=False, header = False)

        return

    def generate_spoken_languages(self, output_file='user_spoken_languages.csv'):
        """
           Creo un CSV che contiene, per ogni utente, la lista di lingue dei tweet con cui ha interagito
        """
        print('Read the training')
        dd_input = dd.read_csv(self.training_file, sep='\u0001', header=None)
        print('Read the training')
        dd_val = self.process_chunk_tsv(dd_input)
        dd_val = dd_val[~dd_val['Reply_engagement_timestamp'].isna() | ~dd_val['Retweet_engagement_timestamp'].isna() | ~dd_val['Retweet_with_comment_engagement_timestamp'].isna() | ~dd_val['Like_engagement_timestamp'].isna()]
        dd_val = dd_input[['User_id_engaging', 'Language']]

        print('Concateno i due df')
        
        dd_val = dd_val.dropna(subset=['Language'])
        dd_val = dd_val[dd_val['Language'] != 0]
        dd_val['Language'] = dd_val['Language'].apply(lambda x: x.split('|') if(isinstance(x, str)) else [])
        dd_val = dd_val.explode('Language')
        dd_val = dd_val.groupby('User_id_engaging')['Language'].apply(list).reset_index().compute()
        dd_val.columns = ['User_id', 'Languages_spoken']
        print(dd_val.head())
        
        dd_val.to_csv(output_file, index=False)
        print('Ho calcolato le lingue parlate da {} utenti'.format(dd_val.shape[0]))
        return

    def generate_user_interaction(self, output_path='./'):

        """
            Genero un csv con il seguente formato:
            User | Lista di autori a cui ha dato un Like | Retweet | Comment | Reply
        """
        label_cols = ['Reply_engagement_timestamp', 'Retweet_engagement_timestamp', 'Retweet_with_comment_engagement_timestamp', 'Like_engagement_timestamp']
    
        dd_input = dd.read_csv(self.training_file, sep='\u0001', header=None)
        dd_input = self.process_chunk_tsv(dd_input)
        
        for label in label_cols:
            print('Compute interaction for label {}'.format(label))
            dd_label = dd_input[['User_id_engaging', 'User_id', label]]
            # Prendo solo quelle che hanno la label non nulla
            dd_label = dd_label[~dd_label[label].isna()]
            df_result = dd_label.groupby('User_id_engaging')['User_id'].apply(list).reset_index().compute()
            df_result.columns = ['User', label.replace('engagement_timestamp', 'interactions')]
            print('Label {} has #{} users'.format(label.replace('_engagement_timestamp', ''), df_result.shape[0]))
            df_result.to_csv(os.path.join(output_path,'user_interactions_{}.csv'.format(label.replace('_engagement_timestamp', ''))), index = False)
            del df_result
            gc.collect()
        
        return

    def generate_user_hashtag(self):
        """
        Genero un csv con il seguente formato:
        User | Lista di hashtag a cui ha dato un Like | Retweet | Comment | Reply
        """

        print('Genero la lista sul training')
        label_cols = ['Reply_engagement_timestamp', 'Retweet_engagement_timestamp', 'Retweet_with_comment_engagement_timestamp', 'Like_engagement_timestamp']
        
        dd_input = dd.read_csv(self.training_file, sep='\u0001', header=None)
        dd_input = self.process_chunk_tsv(dd_input)
        dd_input = dd_input[['User_id_engaging', 'Hashtags'] + label_cols]

        for label in label_cols:
            print('Compute interaction for label {}'.format(label))
            dd_label = dd_input[['User_id_engaging', 'Hashtags', label]]
            # Prendo solo quelle che hanno la label non nulla
            dd_label = dd_label[~dd_label[label].isna()]
            dd_label = dd_label[['User_id_engaging', 'Hashtags']]

            df_result = dd_label.groupby('User_id_engaging')['Hashtags'].apply(list).reset_index().compute()
            df_result.columns = ['User', label.replace('engagement_timestamp', 'hashtags')]
            print('Label {} has #{} users'.format(label.replace('_engagement_timestamp', ''), df_result.shape[0]))
            df_result.to_csv('user_hashtags_{}.csv'.format(label.replace('_engagement_timestamp', '')), index = False)
            del df_result
            gc.collect()
        return

    def count_tweet_interactions(self):
        df = dd.read_csv(self.training_file, sep='\u0001', header=None)
        df = self.process_chunk_tsv(df)
        label_cols = ['Reply_engagement_timestamp', 'Retweet_engagement_timestamp', 'Retweet_with_comment_engagement_timestamp', 'Like_engagement_timestamp']

        df = df[['Tweet_id'] + label_cols]
        df = df[~df['Reply_engagement_timestamp'].isna() | ~df['Retweet_engagement_timestamp'].isna() | ~df['Retweet_with_comment_engagement_timestamp'].isna() | ~df['Like_engagement_timestamp'].isna()]
        df = df.fillna(0)
        for l in label_cols:
            df.loc[:, l] = df[l].apply(lambda x: 1 if x > 0 else 0)
        df = df.groupby('Tweet_id').agg({ 'Reply_engagement_timestamp': 'sum',
                                                                            'Retweet_engagement_timestamp': 'sum',
                                                                            'Retweet_with_comment_engagement_timestamp': 'sum',
                                                                            'Like_engagement_timestamp': 'sum'
                                                                        }).reset_index().compute()
        df.to_csv('tweet_interactions.csv', index=False)
        return

    def count_author_interactions(self, output_file):
        df = dd.read_csv(self.training_file, sep='\u0001', header=None)
        df = self.process_chunk_tsv(df)
        label_cols = ['Reply_engagement_timestamp', 'Retweet_engagement_timestamp', 'Retweet_with_comment_engagement_timestamp', 'Like_engagement_timestamp']

        df = df[['User_id', 'Tweet_id'] + label_cols]
        df = df[~df['Reply_engagement_timestamp'].isna() | ~df['Retweet_engagement_timestamp'].isna() | ~df['Retweet_with_comment_engagement_timestamp'].isna() | ~df['Like_engagement_timestamp'].isna()]
        df = df.fillna(0)
        df['Reply_engagement_timestamp'] = df['Reply_engagement_timestamp'].apply(lambda x: 1 if x > 0 else 0)
        df['Retweet_engagement_timestamp'] = df['Retweet_engagement_timestamp'].apply(lambda x: 1 if x > 0 else 0)
        df['Retweet_with_comment_engagement_timestamp'] = df['Retweet_with_comment_engagement_timestamp'].apply(lambda x: 1 if x > 0 else 0)
        df['Like_engagement_timestamp'] = df['Like_engagement_timestamp'].apply(lambda x: 1 if x > 0 else 0)
        print('Calcolo la size di ogni gruppo')
        res = df[['User_id', 'Tweet_id']].drop_duplicates().compute()
        print(res)
        print(res.shape)
        res = res['User_id'].value_counts().reset_index()
        print(res)
        res.columns = ['User_id', 'ntweets']
        print(res.head())
        print('Tot_tweets {}'.format(res.shape))
        df_out = df[['User_id'] + label_cols].groupby('User_id').agg({ 'Reply_engagement_timestamp': 'sum',
                                                                            'Retweet_engagement_timestamp': 'sum',
                                                                            'Retweet_with_comment_engagement_timestamp': 'sum',
                                                                            'Like_engagement_timestamp': 'sum'
                                                                        }).reset_index().compute()
       
        df_out = df_out.merge(res, left_on=['User_id'], right_on=['User_id'])
        print(df_out.head())

        for l in label_cols:
            df_out.loc[:,l.replace('engagement_timestamp', 'tweet_ratio')] = df_out[l]/df_out['ntweets']
        df_out.to_csv(output_file, index=False)
        return
        
    def get_all_tweet(self, isVal, validation_file='/mnt/val.tsv'):
        if(isVal):
            dd_input = dd.read_csv(validation_file, sep='\u0001', header=None)
            dd_input.columns = self.col_names_val
        else:
            dd_input = dd.read_csv(self.training_file, sep='\u0001', header=None)
            dd_input.columns = self.col_names_training
        n_tweets = dd_input['Tweet_id'].unique().compute()
        self.print_and_log('The Dataframe has {} tweets'.format(n_tweets.shape[0]))
        return set(n_tweets)

    def tweet_in_val_and_training(self):
        training_tweet = self.get_all_tweet(isVal=False)
        self.print_and_log('There are {} tweet in training'.format(len(training_tweet)))
        validation_tweet = self.get_all_tweet(isVal=True)
        self.print_and_log('There are {} tweet in validation'.format(len(validation_tweet)))

        double_tweet = training_tweet.intersection(validation_tweet)
        self.print_and_log('{} tweet are both in training and validation'.format(len(double_tweet)))
        
        return