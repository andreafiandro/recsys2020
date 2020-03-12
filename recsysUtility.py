import pandas as pd
import os 
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import numpy as np
import gc
import time
import matplotlib.pyplot as plt


class RecSysUtility:

   
    def __init__(self, training_file):
        self.training_file = training_file

        self.col_names_val = ['Text tokens', 'Hashtags', 'Tweet id', 'Present media', 'Present links', 'Present domains', 'Tweet type', 'Language', 'Timestamp',
        'User id', 'Follower count', 'Following count', 'Is verified', 'Account creation time',
        'User id engaging', 'Follower count engaging', 'Following count engaging', 'Is verified engaging', 'Account creation time engaging',
        'Engagee follows engager']

        self.col_names_training = ['Text tokens', 'Hashtags', 'Tweet id', 'Present media', 'Present links', 'Present domains', 'Tweet type', 'Language', 'Timestamp',
        'User id', 'Follower count', 'Following count', 'Is verified', 'Account creation time',
        'User id engaging', 'Follower count engaging', 'Following count engaging', 'Is verified engaging', 'Account creation time engaging',
        'Engagee follows engager', 'Reply engagement timestamp', 'Retweet engagement timestamp', 'Retweet with comment engagement timestamp', 'Like engagement timestamp']



    def reduce_mem_usage(self, df):
        """ 
        iterate through all the columns of a dataframe and 
        modify the data type to reduce memory usage.        
        """
        start_mem = df.memory_usage().sum() / 1024**2
        print(('Memory usage of dataframe is {:.2f}' 
                        'MB').format(start_mem))
        
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type != object:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max <\
                    np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max <\
                    np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max <\
                    np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max <\
                    np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)  
                else:
                    if c_min > np.finfo(np.float16).min and c_max <\
                    np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max <\
                    np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
            else:
                df[col] = df[col].astype('category')
        end_mem = df.memory_usage().sum() / 1024**2
        print(('Memory usage after optimization is: {:.2f}' 
                                'MB').format(end_mem))
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) 
                                                / start_mem))
    
        return df

    def clean_col(self, x):
        """
            Used to clean the columns separated by \t to make them readable
        """
        if(pd.isna(x)):
            return x
        else:
            return x.replace('\t', '|')
    

    def process_chunk_tsv(self, df, col_to_clean=['Text tokens', 'Hashtags', 'Present media', 'Present links', 'Present domains'], isVal=False):
        if(isVal):
            df.columns = self.col_names_val
        else:
            df.columns = self.col_names_training

        # Convert boolean to 1 / 0
        df['Is verified'] = df['Is verified'].apply(lambda  x: 1 if x else 0)
        df['Is verified engaging'] = df['Is verified engaging'].apply(lambda  x: 1 if x else 0)
        
        for col in col_to_clean:
            df[col] = df[col].apply(lambda x: self.clean_col(x))
            df[col] = df[col].fillna(0)
        return df

    def create_chunk_csv(self, output_dir='./training', chunk_size = 1000000):

        """
            This functions take the original training.tsv file provided by the organizers of the challenge and generate n smaller files.
            INPUT:
                - output_dir -> Path of the directory of the output files
                - chunk_size -> Number of rows of each chunk
            OUTPUT:
                - write files on disk with name training_chunk_n.csv
        """
        
        chunk_n = 0
        for chunk in pd.read_csv(self.training_file, sep='\u0001', header=None, chunksize=chunk_size):
            df_chunk = self.process_chunk_tsv(chunk)
            df_chunk.to_csv(os.path.join(output_dir, 'training_chunk_{}.csv'.format(chunk_n)), index=False)
            chunk_n += 1

    def count_item(self, x):
        if(x != 0):
            return len(x.split('|'))


    
    
    def generate_submission(self, validation_file, label):
        """
            Function used to generate the submission file.
            Starting from the file 
        """

        not_useful_cols = ['Tweet id', 'User id', 'User id engaging']
        id = 0
        for val in pd.read_csv(validation_file, sep='\u0001', header=None, chunksize=3000000):
            print('Predicting chunk {}'.format(id))
            val = self.process_chunk_tsv(val, isVal=True)
            df_out = pd.DataFrame(columns = ['Tweet_id', 'User_id', 'Prediction'])
            df_out['Tweet_id'] = val['Tweet id']
            df_out['User_id'] = val['User id engaging']
            print('Starting feature engineering...')
            val = self.generate_features_lgb(val)
            val = self.encode_string_features(val)
            val = val.drop(not_useful_cols, axis=1)

            print('Load LGBM model')
            bst = lgb.Booster(model_file='model_Like.txt')
            print('Start Prediction')
            df_out['Prediction'] = bst.predict(val)
            df_out.to_csv('prediction_{}_{}.csv'.format(label, id), index=False, header=False)
            id += 1
            del val, df_out
            gc.collect()

    def incremental_gradient_boosting(self, label):
        """
            This function is used to train a gradient boosting model by means of incremental learning.
            INPUT:
                - label -> the label for the training model (Like, Retweet, Comment or Reply)
            OUTPUT:
                - trained lgbm model that will be also written on the disk
        """      
        label = label + ' engagement timestamp'
        lgb_estimator = None
        params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': {'auc'},
            'num_leaves': 170,
            'n_estimators': 100,
            'learning_rate': 0.01,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0
        }
        not_useful_cols = ['Tweet id', 'User id', 'User id engaging', 'Reply engagement timestamp', 'Retweet engagement timestamp', 'Retweet with comment engagement timestamp', 'Like engagement timestamp']
        
        for df_chunk in pd.read_csv(self.training_file, sep='\u0001', header=None, chunksize=5000000):
            print('Processing the chunk...')
            df_chunk = self.process_chunk_tsv(df_chunk)
            print('Starting feature engineering...')
            df_chunk = self.generate_features_lgb(df_chunk)
            df_chunk = self.encode_string_features(df_chunk)


            print('Split training and test set')
            df_train, df_val = train_test_split(df_chunk, test_size=0.1)   
            print('Training size: {}'.format(df_train.shape[0]))
            print('Validation size: {}'.format(df_val.shape[0]))


            print('Removing column not useful from training')
            y_train = df_train[label].fillna(0)
            y_train = y_train.apply(lambda x : 0 if x == 0 else 1)
            X_train = df_train.drop(not_useful_cols, axis=1)
            y_val = df_val[label].fillna(0)
            y_val = y_val.apply(lambda x : 0 if x == 0 else 1)
            X_val = df_val.drop(not_useful_cols, axis=1)
            print(X_val.head())
            #print(y_val)

            print('Start training...')
            lgb_estimator = lgb.train(params,
                        keep_training_booster=True,
                        # Pass partially trained model:
                        init_model=lgb_estimator,
                        train_set=lgb.Dataset(X_train, y_train),
                        valid_sets=lgb.Dataset(X_val, y_val),
                        num_boost_round=10)
            del df_chunk, X_train, y_train, X_val, y_val
            gc.collect()

        lgb_estimator.save_model('model_{}.txt'.format(label))
        lgb.plot_importance(lgb_estimator, importance_type='split', max_num_features=50)
        lgb.plot_importance(lgb_estimator, importance_type='gain', max_num_features=50)
        ax = lgb.plot_tree(lgb_estimator, figsize=(15, 15), show_info=['split_gain'])
        plt.show()
        return lgb_estimator



    def generate_features_lgb(self, df):
        """
        Function to generate the features included in the gradient boosting model.
        """

        # Count the number of the items in the following columns
        # TODO: Present link, media e domains per ora sono semplicemente sommati, in realtà andrebbe specificato il tipo di media con il numero di elementi (es. 2 video, 3 foto ecc...)
        col_to_count=['Text tokens', 'Hashtags', 'Present media', 'Present links', 'Present domains']

        for col in col_to_count:
            df[col] = df[col].apply(lambda x: self.count_item(x))

        # Instead of timestamp, I count the seconds elapsed from the account creation
        current_timestamp = int(time.time())
        df['Account creation time'] = df['Account creation time'].apply(lambda x: current_timestamp - x)
        df['Account creation time engaging'] = df['Account creation time engaging'].apply(lambda x: current_timestamp - x)
        
        return df

    def encode_string_features(self, df):
        """
        Function used to convert the features represented by strings. 
        """
        self.tweet_type_dic = {}
        tweet_type_id = 0
        for t in df['Tweet type'].unique():
            self.tweet_type_dic[t] = tweet_type_id
            tweet_type_id += 1

        self.lang_dic = {}
        lang_id = 0
        for t in df['Language'].unique():
            self.lang_dic[t] = lang_id
            lang_id += 1

        df['Tweet type'] = df['Tweet type'].apply(lambda x: self.tweet_type_dic[x])
        df['Language'] = df['Language'].apply(lambda x: self.lang_dic[x])
        return df