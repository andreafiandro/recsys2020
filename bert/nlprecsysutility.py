import pandas as pd
import os 
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, log_loss
import logging
import time
import json

# useful utility functions by AF e SL

class RecSysUtility:

   
    def __init__(self, test_file):
        self.test_file = test_file 
        self.training_file = test_file #For usage in training like xgb
        logging.basicConfig(filename='statistics.log',level=logging.INFO)
        #ProgressBar().register()

        self.col_names_val = ['Text_tokens', 'Hashtags', 'Tweet_id', 'Present_media', 'Present_links', 'Present_domains', 'Tweet_type', 'Language', 'Timestamp',
        'User_id', 'Follower_count', 'Following_count', 'Is_verified', 'Account_creation_time',
        'User_id_engaging', 'Follower_count_engaging', 'Following_count_engaging', 'Is_verified_engaging', 'Account_creation_time_engaging',
        'Engagee_follows_engager']

        self.col_names_training = ['Text_tokens', 'Hashtags', 'Tweet_id', 'Present_media', 'Present_links', 'Present_domains', 'Tweet_type', 'Language', 'Timestamp',
        'User_id', 'Follower_count', 'Following_count', 'Is_verified', 'Account_creation_time',
        'User_id_engaging', 'Follower_count_engaging', 'Following_count_engaging', 'Is_verified_engaging', 'Account_creation_time_engaging',
        'Engagee_follows_engager', 'Reply_engagement_timestamp', 'Retweet_engagement_timestamp', 'Retweet_with_comment_engagement_timestamp', 'Like_engagement_timestamp']
        
        # Indexes for features
        self.tweet_type_id = 0
        self.lang_id = 0
        self.tweet_type_dic = {}
        self.lang_dic = {}

    """
    ------------------------------------------------------------------------------------------
    UTILITIES
    ------------------------------------------------------------------------------------------
    """

    def clean_col(self, x):
        """
            Used to clean the columns separated by \t to make them readable
        """
        if(pd.isna(x)):
            return x
        else:
            return x.replace('\t', '|')
    

    def process_chunk_tsv(self, df, col_to_clean=['Text_tokens', 'Hashtags', 'Present_media', 'Present_links', 'Present_domains'], isVal=True):
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


    def create_chunk_csv(self, output_dir='./test_chunk', chunk_size = 1000000):

        """
            This functions take the original training.tsv file provided by the organizers of the challenge and generate n smaller files.
            INPUT:
                - output_dir -> Path of the directory of the output files
                - chunk_size -> Number of rows of each chunk
            OUTPUT:
                - write files on disk with name training_chunk_n.csv
        """
        
        chunk_n = 0
        for chunk in pd.read_csv(self.test_file, sep='\u0001', header=None, chunksize=chunk_size):
            df_chunk = self.process_chunk_tsv(chunk)
            df_chunk.to_csv(os.path.join(output_dir, 'test_chunk_{}.csv'.format(chunk_n)), index=False)
            chunk_n += 1


    def count_item(self, x):
        if(x != 0):
            return len(x.split('|'))


    def save_dictionaries_on_file(self):
        if(os.path.exists('lang.json')):
            os.remove('lang.json')

        f = open("lang.json","w")
        f.write(json.dumps(self.lang_dic))
        f.close()

        if(os.path.exists('tweet_type.json')):
            os.remove('tweet_type.json')
            
        f = open("tweet_type.json","w")
        f.write(json.dumps(self.tweet_type_dic))
        f.close()
        return


    def encode_string_features(self, df, isDask=False):
        """
        Function used to convert the features represented by strings. 
        """

        # Aggiorno i dizionari
        for t in df['Tweet_type'].unique():
            if t not in self.tweet_type_dic:
                self.tweet_type_dic[t] = self.tweet_type_id
                self.tweet_type_id += 1

        

        for t in df['Language'].unique():
            if t not in self.lang_dic:
                self.lang_dic[t] = self.lang_id
                self.lang_id += 1
        
        # Salvo i dizionari su json
        self.save_dictionaries_on_file()
        if(isDask):
            df['Tweet_type'] = df['Tweet_type'].apply(lambda x: self.tweet_type_dic[x], meta=('int'))
            df['Language'] = df['Language'].apply(lambda x: self.lang_dic[x], meta=('int'))
        else:
            df['Tweet_type'] = df['Tweet_type'].apply(lambda x: self.tweet_type_dic[x])
            df['Language'] = df['Language'].apply(lambda x: self.lang_dic[x])
        return df


    def encode_val_string_features(self, df):
        """
        Function used to encode the string features by means of the dictionaries generated during the training, useful during submission
        """

        jsonFile = open("lang.json", "r")
        lang_dic = json.load(jsonFile)
        jsonFile.close()

        jsonFile = open("tweet_type.json", "r")
        tweet_type_dic = json.load(jsonFile)
        jsonFile.close()

        df['Tweet_type'] = df['Tweet_type'].apply(lambda x: tweet_type_dic.get(x, -1))
        df['Language'] = df['Language'].apply(lambda x: lang_dic.get(x, -1))

        return df
    

    def generate_features_lgb_mod(self, df, user_features_file='./user_features_final.csv'):
        """
        Function to generate the features included in the gradient boosting model.
        """

        # Count the number of the items in the following columns
        # TODO: Present link, media e domains per ora sono semplicemente sommati, in realtà andrebbe specificato il tipo di media con il numero di elementi (es. 2 video, 3 foto ecc...)
        time_col_to_days = ['Timestamp', 'Account_creation_time', 'Account_creation_time_engaging']
        for col in time_col_to_days:
            df.loc[:, col] = df[col] / 86400
            
        col_to_count=['Hashtags', 'Present_media', 'Present_links', 'Present_domains']

        for col in col_to_count:
            df.loc[:, col] = df[col].apply(lambda x: self.count_item(x))
        df.loc[:, 'Text_len'] = df['Text_tokens'].apply(lambda x: self.count_item(x) -2)

        # Instead of timestamp, I count the days elapsed from the account creation
        current_timestamp = int(time.time()) / 86400
        df.loc[:, 'Account_creation_time'] = df['Account_creation_time'].apply(lambda x: current_timestamp - x)
        df.loc[:, 'Account_creation_time_engaging'] = df['Account_creation_time_engaging'].apply(lambda x: current_timestamp - x)
        
        # Runtime Features
        df.loc[:,"follow_ratio_author"] = df.loc[:,'Following_count'] / (df.loc[:,'Follower_count'] + 1)
        df.loc[:,"follow_ratio_user"] = df.loc[:,'Following_count_engaging'] / (df.loc[:,'Follower_count_engaging'] + 1)
        df.loc[:,'Elapsed_time_author'] = df['Timestamp'] - df['Account_creation_time']
        df.loc[:,'Elapsed_time_user'] = df['Timestamp'] - df['Account_creation_time_engaging']

        # Add user features
        print('Adding the user features from {}'.format(user_features_file))
        df_input = pd.read_csv(user_features_file, nrows=None) 
        df = df.merge(df_input, how='left', left_on='User_id_engaging', right_on='User_id_engaging')
        
        # Create other features
        df.loc[:,'ratio_reply'] = df.loc[:,'Tot_reply'] / (df.loc[:,'Tot_action'] + 1)
        df.loc[:,'ratio_retweet'] = df.loc[:,'Tot_retweet'] / (df.loc[:,'Tot_action'] + 1)
        df.loc[:,'ratio_comment'] = df.loc[:,'Tot_comment'] / (df.loc[:,'Tot_action'] + 1)
        df.loc[:,'ratio_like'] = df.loc[:,'Tot_like'] / (df.loc[:,'Tot_action'] + 1)

        # Riempio i valori NaN con -1 per dare un informazione in più al gradient boosting
        col_to_fill = ['Tot_reply', 'Tot_retweet', 'Tot_comment', 'Tot_like', 'Tot_action', 'ratio_reply', 'ratio_retweet', 'ratio_comment', 'ratio_like']
        df[col_to_fill] = df[col_to_fill].fillna(value=0)

        return df
    

    """
    ------------------------------------------------------------------------------------------
    OFFICIAL FUNCTIONS FOR EVALUATE THE SCORE
    ------------------------------------------------------------------------------------------
    """

    def compute_prauc(self, pred, gt):
        prec, recall, _ = precision_recall_curve(gt, pred)
        prauc = auc(recall, prec)
        return prauc


    def calculate_ctr(self,gt):
        positive = len([x for x in gt if x == 1])
        ctr = positive/float(len(gt))
        return ctr


    def compute_rce(self, pred, gt):
        #pred = np.asarray(pred, dtype=np.float64)
        pred = np.around(pred, decimals=5)
        cross_entropy = log_loss(gt, pred)
        #cross_entropy = log_loss(gt, pred, labels=[0,1])
        data_ctr = self.calculate_ctr(gt)
        strawman_cross_entropy = log_loss(gt, [data_ctr for _ in range(len(gt))])
        return (1.0 - cross_entropy/strawman_cross_entropy)*100.0


    # def compute_rce_xgb(self, pred, gt):
    #     #gt = np.asarray(gt.get_label(), dtype=np.int64)
    #     #pred = np.asarray(pred, dtype=np.float64)
    #     pred = np.around(pred, decimals=5)
    #     cross_entropy = log_loss(gt, pred)
    #     #cross_entropy = log_loss(gt, pred, labels=[0,1])
    #     data_ctr = self.calculate_ctr(gt)
    #     strawman_cross_entropy = log_loss(gt, [data_ctr for _ in range(len(gt))], labels=[0,1])
    #     return 'RCE', (1.0 - cross_entropy/strawman_cross_entropy)*100.0

    def print_and_log(self, to_print):
        logging.info(to_print)
        print(to_print)
        return

# utility functions by IO

def submission_files(submission_dataframe, output_dir="./"):
        submission_dataframe.to_csv(os.path.join(output_dir,'reply_prediction.csv'),columns = ['Tweet_Id','User_Id','Reply'],index=False, header = False)
        submission_dataframe.to_csv(os.path.join(output_dir,'retweet_prediction.csv'),columns = ['Tweet_Id','User_Id','Retweet'],index=False, header = False)
        submission_dataframe.to_csv(os.path.join(output_dir,'retweet_with_comment_prediction.csv'),columns = ['Tweet_Id','User_Id','Retweet_with_comment'],index=False , header = False)
        submission_dataframe.to_csv(os.path.join(output_dir,'like_prediction.csv'),columns = ['Tweet_Id','User_Id','Like'],index=False, header = False)