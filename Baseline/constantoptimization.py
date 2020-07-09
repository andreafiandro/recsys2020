import dask.dataframe as dd
import logging
from dask.diagnostics import ProgressBar
import pandas as pd


class ConstantOptimization:

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
        Constant Optimization
    ------------------------------------------------------------------------------------------
    """

    def count_n_lines(self):
        dd_input = dd.read_csv(self.training_file, sep='\u0001', header=None)
        n_rows = dd_input.shape[0].compute()
        self.print_and_log('The Dataframe has {} lines'.format(n_rows))
        return n_rows

    def count_action_type(self, label, isVal=False):
        print('I get all the {} actions'.format(label))
        label_pandas = label + '_engagement_timestamp'
        dd_input = dd.read_csv(self.training_file, sep='\u0001', header=None)
        dd_input = self.process_chunk_tsv(dd_input, isVal=isVal)
        df_not_null = dd_input[label_pandas][~dd_input[label_pandas].isna()].compute()
        self.print_and_log('There are {} action of type {}'.format(df_not_null.shape[0], label))
        return df_not_null.shape[0]
  
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

    def optimize_constant(self, validation_file, filename, compute_ctr=False):

        if compute_ctr:
            tot_lines = self.count_n_lines()
            ctr_like = self.count_action_type('Like')
            ctr_reply = self.count_action_type('Reply')
            ctr_retweet = self.count_action_type('Retweet')
            ctr_comment = self.count_action_type('Retweet_with_comment')

        else:
            tot_lines = 123126259
            ctr_like = 52719548 / tot_lines
            ctr_reply = 3108287 / tot_lines
            ctr_retweet = 13264420 / tot_lines
            ctr_comment = 884082 / tot_lines

        df_val = pd.read_csv(validation_file, sep='\u0001', header=None)
        df_val = self.process_chunk_tsv(df_val, isVal=True)
        df_val = df_val[['Tweet_id', 'User_id_engaging']]

        df_val.loc[:, 'prediction_Like'] = ctr_like
        df_val.loc[:, 'prediction_Reply'] = ctr_reply
        df_val.loc[:, 'prediction_Retweet'] = ctr_retweet
        df_val.loc[:, 'prediction_Comment'] = ctr_comment

        lista_label = ['Reply', 'Retweet', 'Comment', 'Like']
        for l in lista_label:
            df_val[['Tweet_id', 'User_id_engaging', 'prediction_{}'.format(l)]].to_csv(
                'prediction_{}_{}.csv'.format(filename, l), header=None, index=False)

        return