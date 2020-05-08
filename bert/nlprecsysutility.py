import pandas as pd
import os 
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, log_loss
import logging

# useful utility functions by AF e SL

class RecSysUtility:

   
    def __init__(self, test_file):
        self.test_file = test_file
        logging.basicConfig(filename='statistics.log',level=logging.INFO)
        ProgressBar().register()

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


    
    

    """
    ------------------------------------------------------------------------------------------
    OFFICIAL FUNCTIONS FOR EVALUATE THE SCORE
    ------------------------------------------------------------------------------------------
    """

    def compute_prauc(self, pred, gt):
        prec, recall, thresh = precision_recall_curve(gt, pred)
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