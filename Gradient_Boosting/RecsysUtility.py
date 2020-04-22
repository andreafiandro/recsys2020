import pandas as pd
import os 
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from dask.diagnostics import ProgressBar
import numpy as np
import gc
import time
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, log_loss
import logging
import dask
import xgboost as xgb
import json
import pickle
from RecsysStats import RecSysStats

class RecSysUtility:

   
    def __init__(self, training_file):
        self.training_file = training_file
        logging.basicConfig(filename='training.log',level=logging.INFO)
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

        if(os.path.exists('col_name')):
            f = open("col_name", "r")
            self.name_of_features = f.readline().split(',')   
        else:
            self.name_of_features = []

    def generate_submission(self, validation_file, label, gb_type='xgb'):
        """
            Function used to generate the submission file.
            Starting from the file 
        """

        if not os.path.exists(label):
            os.system('mkdir {}'.format(label))

        not_useful_cols = ['Tweet_id', 'User_id', 'User_id_engaging']
        id = 0
        for val in pd.read_csv(validation_file, sep='\u0001', header=None, chunksize=5000000):
            print('Predicting chunk {}'.format(id))
            val = self.process_chunk_tsv(val, isVal=True)
            df_out = pd.DataFrame(columns = ['Tweet_id', 'User_id', 'Prediction'])
            df_out['Tweet_id'] = val['Tweet_id']
            df_out['User_id'] = val['User_id_engaging']
            print('Starting feature engineering...')
            val = self.generate_features_lgb(val)
            val = self.encode_val_string_features(val)
            val = val.drop(not_useful_cols, axis=1)

            print('Load GB model')

            if(gb_type=='lgbm'):
                model = lgb.Booster(model_file='model_{}.txt'.format(label))
                print('Start Prediction')
                df_out['Prediction'] = model.predict(val)
            elif(gb_type=='xgb'):
                model = pickle.load(open('model_xgb_{}.dat'.format(label), "rb"))

                print('Start Prediction')
                df_out['Prediction'] = model.predict(xgb.DMatrix(val), ntree_limit=model.best_ntree_limit)
            df_out.to_csv('./{}/prediction_{}.csv'.format(label, id), index=False, header=False)
            id += 1
            del val, df_out
            gc.collect()
        
        files_csv = os.listdir('./{}'.format(label))
        to_concat = []
        for f in files_csv:
            to_concat.append(pd.read_csv('./{}/{}'.format(label,f), header=None))
        df_grouped = pd.concat(to_concat, axis=0, ignore_index=True)
        df_grouped.to_csv('prediction_{}.csv'.format(label), index=False)

    def xgboost_training_memory(self, label, training_folder='/datadrive/xgb/'):
        """
            This function is used to train a gradient boosting model by means of incremental learning.
            INPUT:
                - label -> the label for the training model (Like, Retweet, Comment or Reply) 
            OUTPUT:
                - trained lgbm model that will be also written on the disk
        """     
        # Da rimuovere
        self.name_of_features.remove('label')

        print('Pulisco le run precedenti') 
        for a in ['train', 'val', 'test']:
            if(os.path.exists('d{}.cache'.format(a))):
                os.remove('d{}.cache'.format(a))
            if(os.path.exists('d{}.cache.ellpack.page'.format(a))):
                os.remove('d{}.cache.ellpack.page'.format(a))
            if(os.path.exists('d{}.cache.row.page'.format(a))):
                os.remove('d{}.cache.row.page'.format(a))

        estimator = None
        xgb_params = {
            'eta':0.1, 
            'tree_method': 'gpu_hist',
            'sampling_method': 'gradient_based',
            'objective': 'binary:logistic',
            'nthread':6,  
            'seed':1,
            'disable_default_eval_metric': 1
        }
        training_set = xgb.DMatrix('{}training_{}.csv?format=csv&label_column=0#dtrain.cache'.format(training_folder, label), feature_names=self.name_of_features)
        #val_set = xgb.DMatrix('{}validation_{}.csv?format=csv&label_column=0#cacheprefix'.format(training_folder, label))
        val_set = xgb.DMatrix('{}validation_{}.csv?format=csv&label_column=0#dval.cache'.format(training_folder, label), feature_names=self.name_of_features)
        evallist = [(val_set, 'eval'), (training_set, 'train')]

        print('Start training for label {}...'.format(label))

        estimator = xgb.train(xgb_params,
                                num_boost_round=30,
                                early_stopping_rounds=10,
                                feval=self.compute_rce_xgb,
                                maximize=True, 
                                dtrain=training_set,
                                evals=evallist)
        print('Training finito')
        test_set = xgb.DMatrix('{}test_{}.csv?format=csv&label_column=0#dtest.cache'.format(training_folder, label), feature_names=self.name_of_features)
        y_pred = estimator.predict(test_set)
        prauc = self.compute_prauc(y_pred, test_set.get_label())
        rce = self.compute_rce(y_pred, test_set.get_label())

        self.print_and_log('Training for {} --- PRAUC: {} / RCE: {}'.format(label, prauc, rce))
            
        print('Saving model...')
        pickle.dump(estimator, open('model_xgb_{}.dat'.format(label), "wb"))

        ax = xgb.plot_importance(estimator)
        ax.figure.set_size_inches(10,8)
        ax.figure.savefig('importance_{}.png'.format(label))

        return estimator
    
    
    """
    ------------------------------------------------------------------------------------------
    FEATURE GENERATION
    ------------------------------------------------------------------------------------------
    """

    def generate_user_features(self, output_file='user_features.csv'):
        """
        Funzione utile per generare un file che contiene | User_id | Tot_like | ...
        """
        firstFile = True
        tot_lines = 0
        for df_chunk in pd.read_csv(self.training_file, sep='\u0001', header=None, chunksize=6000000):
            print('Analizzo il chunk..')
            df_training = self.process_chunk_tsv(df_chunk)
            df_count_like = df_training[['User_id_engaging','Reply_engagement_timestamp', 'Retweet_engagement_timestamp', 'Retweet_with_comment_engagement_timestamp', 'Like_engagement_timestamp']]
            df_count_like = df_count_like[(~df_count_like['Like_engagement_timestamp'].isnull()) | 
                                        (~df_count_like['Reply_engagement_timestamp'].isnull()) |
                                        (~df_count_like['Retweet_engagement_timestamp'].isnull()) |
                                        (~df_count_like['Retweet_with_comment_engagement_timestamp'].isnull())]
            df_count_like.loc[:,'Reply_engagement_timestamp'] = df_count_like.loc[:,'Reply_engagement_timestamp'].apply(lambda x: 0 if pd.isna(x) else 1)
            df_count_like.loc[:,'Retweet_engagement_timestamp'] = df_count_like.loc[:,'Retweet_engagement_timestamp'].apply(lambda x: 0 if pd.isna(x) else 1)
            df_count_like.loc[:,'Retweet_with_comment_engagement_timestamp'] = df_count_like.loc[:,'Retweet_with_comment_engagement_timestamp'].apply(lambda x: 0 if pd.isna(x) else 1)
            df_count_like.loc[:,'Like_engagement_timestamp'] = df_count_like.loc[:,'Like_engagement_timestamp'].apply(lambda x: 0 if pd.isna(x) else 1)
            df_count_like = df_count_like.groupby('User_id_engaging').agg({ 'Reply_engagement_timestamp': 'sum',
                                                                            'Retweet_engagement_timestamp': 'sum',
                                                                            'Retweet_with_comment_engagement_timestamp': 'sum',
                                                                            'Like_engagement_timestamp': 'sum'
                                                                        }).reset_index()
            df_count_like = df_count_like.rename({'Like_engagement_timestamp': 'Tot_like',
                                      'Reply_engagement_timestamp': 'Tot_reply',
                                     'Retweet_engagement_timestamp': 'Tot_retweet',
                                     'Retweet_with_comment_engagement_timestamp': 'Tot_comment'}, axis='columns')
            df_count_like.loc[:,'Tot_action'] = df_count_like.loc[:,'Tot_like'] + df_count_like.loc[:,'Tot_reply'] + df_count_like.loc[:,'Tot_retweet'] + df_count_like.loc[:,'Tot_comment']
            tot_lines += df_count_like.shape[0]
            print('Ci sono {} utenti'.format(df_count_like.shape[0]))
            print('Le righe totali del file sono {}'.format(tot_lines))
            if(firstFile):
                df_count_like.to_csv(output_file,  index=False)
                firstFile = False
            else:
                df_count_like.to_csv(output_file, mode='a', header=False,  index=False)
        
        del df_count_like
        del df_chunk
        gc.collect()

        print('GroupBy finale')
        df_output = pd.read_csv(output_file)
        dim_iniziale = df_output.shape[0]
        df_output = df_output.groupby('User_id_engaging').agg({ 'Tot_reply': 'sum',
                                                                'Tot_retweet': 'sum',
                                                                'Tot_comment': 'sum',
                                                                'Tot_like': 'sum',
                                                                'Tot_action': 'sum'
                                                            }).reset_index()
        print('User features ha #{} righe -- Dopo groupby #{} righe'.format(dim_iniziale, df_output.shape[0]))
        df_output.to_csv(output_file.replace('.csv', '') + '_final.csv',  index=False)                                                  

        return
    
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

    def generate_features_lgb(self, df, user_features_file='./user_features_final.csv'):
        """
        Function to generate the features included in the gradient boosting model.
        """

        # Count the number of the items in the following columns
        # TODO: Present link, media e domains per ora sono semplicemente sommati, in realtà andrebbe specificato il tipo di media con il numero di elementi (es. 2 video, 3 foto ecc...)
        col_to_count=['Text_tokens', 'Hashtags', 'Present_media', 'Present_links', 'Present_domains']

        for col in col_to_count:
            df[col] = df[col].apply(lambda x: self.count_item(x))

        # Instead of timestamp, I count the seconds elapsed from the account creation
        current_timestamp = int(time.time())
        df['Account_creation_time'] = df['Account_creation_time'].apply(lambda x: current_timestamp - x)
        df['Account_creation_time_engaging'] = df['Account_creation_time_engaging'].apply(lambda x: current_timestamp - x)
        
        # Runtime Features
        df.loc[:,"follow_ratio_author"] = df.loc[:,'Following_count'] / df.loc[:,'Follower_count']
        df.loc[:,"follow_ratio_user"] = df.loc[:,'Following_count_engaging'] / df.loc[:,'Follower_count_engaging']
        df.loc[:,'Elapsed_time_author'] = df['Timestamp'] - df['Account_creation_time']
        df.loc[:,'Elapsed_time_user'] = df['Timestamp'] - df['Account_creation_time_engaging']

        # Add user features
        print('Adding the user features from {}'.format(user_features_file))
        df_input = pd.read_csv(user_features_file)
        df = df.merge(df_input, how='left', left_on='User_id_engaging', right_on='User_id_engaging')
        
        # Create other features
        df.loc[:,'ratio_reply'] = df.loc[:,'Tot_reply'] / df.loc[:,'Tot_action']
        df.loc[:,'ratio_retweet'] = df.loc[:,'Tot_retweet'] / df.loc[:,'Tot_action']
        df.loc[:,'ratio_comment'] = df.loc[:,'Tot_comment'] / df.loc[:,'Tot_action']
        df.loc[:,'ratio_like'] = df.loc[:,'Tot_like'] / df.loc[:,'Tot_action']

        # Riempio i valori NaN con -1 per dare un informazione in più al gradient boosting
        col_to_fill = ['Tot_reply', 'Tot_retweet', 'Tot_comment', 'Tot_like', 'Tot_action', 'ratio_reply', 'ratio_retweet', 'ratio_comment', 'ratio_like']
        df[col_to_fill] = df[col_to_fill].fillna(value=-1)


        return df

    """
    ------------------------------------------------------------------------------------------
        GENERAZIONE DEI FILE DI TRAINING
    ------------------------------------------------------------------------------------------
    """

    def generate_four_files(self, df_in, training_folder, file_type, balanced):
        
        not_useful_cols = ['Tweet_id', 'User_id', 'User_id_engaging', 'Reply_engagement_timestamp', 'Retweet_engagement_timestamp', 'Retweet_with_comment_engagement_timestamp', 'Like_engagement_timestamp']
        labels = ['Reply_engagement_timestamp', 'Retweet_engagement_timestamp', 'Retweet_with_comment_engagement_timestamp', 'Like_engagement_timestamp']
        for label in labels:
                X_train = df_in.drop(not_useful_cols, axis=1)
                y_train = df_in[label].fillna(0).apply(lambda x : 0 if x == 0 else 1).astype(int)
                X_train['label'] = y_train
                if balanced:
                    df_positive = X_train[X_train['label'] == 1]
                    df_negative = X_train[X_train['label'] == 0]
                    print('Positive sample: #{} / Negative sample: #{}'.format(df_positive.shape[0], df_negative.shape[0]))
                    if (df_negative.shape[0] > df_positive.shape[0]):
                        initial_shape = df_negative.shape[0]
                        df_negative = df_negative.sample(n=df_positive.shape[0], random_state=99)
                        print('Bilancio i negativi da {} a {}'.format(initial_shape, df_negative.shape[0]))
                    elif (df_negative.shape[0] < df_positive.shape[0]):
                        initial_shape = df_positive.shape[0]
                        df_positive = df_positive.sample(n=df_negative.shape[0], random_state=99)
                        print('Bilancio i positivi da {} a {}'.format(initial_shape, df_negative.shape[0]))
                    X_train = pd.concat([df_positive, df_negative], axis=0, ignore_index=True)
                ## Change the order of columns (First the labels)
                cols = X_train.columns.tolist()
                cols = cols[-1:] + cols[:-1]
                X_train = X_train[cols]
                X_train = X_train.fillna(0)
                X_train = self.reduce_mem_usage(X_train)
                X_train.to_csv('{}{}_{}.csv'.format(training_folder, file_type, label.replace('_engagement_timestamp', '')), mode='a', header=False, index=False)
        return X_train.columns.tolist()

    def generate_training_xgboost(self, training_folder='/datadrive/xgb/', val_size=0.1, test_size=0.1, training_lines=100000000, balanced=False):

        """
            This function is used to generate the file ready for the xgboost training
        """      
        rStat = RecSysStats(self.training_file)
        n_chunk = 0
        if(os.path.exists('col_name')):
            os.remove('col_name')

        if(training_lines == None):
            print('Count the total number lines')
            training_lines = rStat.count_n_lines()
            print('There are {} lines'.format(training_lines))

        # Genero Validation Set
        val_rows = int(val_size * training_lines)
        print('Validation Rows: {}'.format(val_rows))
        df_val = pd.read_csv(self.training_file, sep='\u0001', header=None, nrows=val_rows)
        df_val = self.process_chunk_tsv(df_val)

        print('Starting feature engineering...')
        df_val = self.generate_features_lgb(df_val)
        df_val = self.encode_string_features(df_val)
        self.name_of_features = self.generate_four_files(df_val, training_folder, 'validation', balanced)
        self.name_of_features.remove('label')
         # Salvo i nomi delle colonne
        with open("col_name", "w") as outfile:
            outfile.write(",".join(self.name_of_features))
            
        test_rows = int(test_size * training_lines)
        print('Test Rows: {}'.format(val_rows))
        df_test = pd.read_csv(self.training_file, sep='\u0001', header=None, nrows=test_rows, skiprows=val_rows)
        df_test = self.process_chunk_tsv(df_test)

        print('Starting feature engineering...')
        df_test = self.generate_features_lgb(df_test)
        df_test = self.encode_string_features(df_test)
        self.generate_four_files(df_test, training_folder, 'test', balanced)


        for df_chunk in pd.read_csv(self.training_file, sep='\u0001', header=None, chunksize=10000000, skiprows=val_rows+test_rows):
            
            print('Processing the chunk {}...'.format(n_chunk))
            
            n_chunk +=1
            df_chunk = self.process_chunk_tsv(df_chunk)

            print('Starting feature engineering...')

            df_chunk = self.generate_features_lgb(df_chunk)
            df_chunk = self.encode_string_features(df_chunk)
            self.generate_four_files(df_chunk, training_folder, 'training', balanced)
        return 

    def generate_dictionary(self):
        n_chunk = 0
        for df_chunk in pd.read_csv(self.training_file, sep='\u0001', header=None, chunksize=10000000):
                    
            print('Processing the chunk {}...'.format(n_chunk))
            n_chunk +=1
            df_chunk = self.process_chunk_tsv(df_chunk)
            print('Starting feature engineering...')
            df_chunk = self.encode_string_features(df_chunk)

    def evaluate_saved_model(self, label, training_folder='/datadrive/xgb/'):
        not_useful_cols = ['Tweet_id', 'User_id', 'User_id_engaging']
        model = pickle.load(open('model_xgb_{}.dat'.format(label), "rb"))
        test_set = xgb.DMatrix('{}test_{}.csv?format=csv&label_column=0'.format(training_folder, label))
        y_pred = model.predict(test_set, ntree_limit=model.best_ntree_limit)
        prauc = self.compute_prauc(y_pred, test_set.get_label())
        rce = self.compute_rce(y_pred, test_set.get_label())
        self.print_and_log('Training for {} --- PRAUC: {} / RCE: {}'.format(label, prauc, rce))
        return

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
        print('Data CTR: {} / Strawman: {} / Cross entropy: {}'.format(data_ctr, strawman_cross_entropy, cross_entropy))
        return (1.0 - cross_entropy/strawman_cross_entropy)*100.0

    def compute_rce_xgb(self, pred, gt):
        gt = np.asarray(gt.get_label(), dtype=np.int64)
        pred = np.asarray(pred, dtype=np.float64)
        #cross_entropy = log_loss(gt, pred)
        cross_entropy = log_loss(gt, pred, labels=[0,1])
        data_ctr = self.calculate_ctr(gt)
        strawman_cross_entropy = log_loss(gt, [data_ctr for _ in range(len(gt))], labels=[0,1])
        return 'RCE', (1.0 - cross_entropy/strawman_cross_entropy)*100.0

   
    """
    ------------------------------------------------------------------------------------------
    UTILITIES
    ------------------------------------------------------------------------------------------
    """


    def print_and_log(self, to_print):
        logging.info(to_print)
        print(to_print)
        return

    def reduce_mem_usage(self, df):
        """ 
        NON FUNZIONA
        iterate through all the columns of a dataframe and 
        modify the data type to reduce memory usage.        
        """
        start_mem = df.memory_usage().sum() / 1024**2
        self.print_and_log(('Memory usage of dataframe is {:.2f}' 
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
        self.print_and_log(('Memory usage after optimization is: {:.2f}' 
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
    
    def exploit_leaks_retweet(self, submission_file, solution_file):
        df_sol = pd.read_csv(solution_file)
        df_sol.columns = ['user_id', 'tweet_id', 'prediction']
        df_sol = df_sol[['User_id_attuale', 'Tweet_id']]
        df_sub = pd.read_csv(submission_file)
        df_sub = df_sub.merge(df_sol, left_on=['user_id', 'tweet_id'], right_on=['User_id_attuale', 'Tweet_id'])
        print(df_sub.head())
        return

        
