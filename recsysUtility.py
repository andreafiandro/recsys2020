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
from sklearn.metrics import precision_recall_curve, auc, log_loss
import logging
from dask.distributed import Client, LocalCluster
from dask_ml.model_selection import train_test_split as dask_split
import dask
import xgboost as xgb
import dask_xgboost
import json
import pickle

class RecSysUtility:

   
    def __init__(self, training_file):
        self.training_file = training_file
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


    def generate_submission(self, validation_file, label, gb_type='xgb'):
        """
            Function used to generate the submission file.
            Starting from the file 
        """

        not_useful_cols = ['Tweet_id', 'User_id', 'User_id_engaging']
        id = 0
        for val in pd.read_csv(validation_file, sep='\u0001', header=None, chunksize=3000000):
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
                df_out['Prediction'] = model.predict(xgb.DMatrix(val))
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

    def incremental_gradient_boosting(self, label, type_gb='xgb'):
        """
            This function is used to train a gradient boosting model by means of incremental learning.
            INPUT:
                - label -> the label for the training model (Like, Retweet, Comment or Reply)
            OUTPUT:
                - trained lgbm model that will be also written on the disk
        """      
        label = label + '_engagement_timestamp'
        estimator = None
        if(type_gb=='lgbm'):
            lgbm_params = {
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
        elif(type_gb=='xgb'):
            xgb_params = {
                'eta':0.1, 
                'booster':'gbtree',
                'max_depth':7,         
                'nthread':4,  
                'seed':1
            }


        not_useful_cols = ['Tweet_id', 'User_id', 'User_id_engaging', 'Reply_engagement_timestamp', 'Retweet_engagement_timestamp', 'Retweet_with_comment_engagement_timestamp', 'Like_engagement_timestamp']
        n_chunk = 0
        first_file = True
        for df_chunk in pd.read_csv(self.training_file, sep='\u0001', header=None, chunksize=7000000):
            print('Processing the chunk...')
            df_chunk = self.process_chunk_tsv(df_chunk)
            #df_negative = df_chunk[df_chunk[label].isna()]
            #df_positive = df_chunk[~df_chunk[label].isna()]
            #n_positive = df_chunk[label].isna()
            #print('Positive sample: #{} / Negative sample: #{}'.format(df_positive.shape[0], df_negative.shape[0]))
            #df_negative = df_negative.sample(n=df_positive.shape[0], random_state=1)
            #print('RESAMPLE -- Positive sample: #{} / Negative sample: #{}'.format(df_positive.shape[0], df_negative.shape[0]))
            print('Starting feature engineering...')
            #df_chunk = pd.concat([df_positive, df_negative], axis=0, ignore_index=True)
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
            if(type_gb=='lgbm'):
                estimator = lgb.train(lgbm_params,
                            keep_training_booster=True,
                            # Pass partially trained model:
                            init_model=estimator,
                            train_set=lgb.Dataset(X_train, y_train),
                            valid_sets=lgb.Dataset(X_val, y_val),
                            num_boost_round=10)
            elif(type_gb=='xgb'):
                if(first_file):
                    estimator = xgb.train(xgb_params, 
                                          num_boost_round=100,
                                          early_stopping_rounds=30,
                                          feval=self.compute_rce_xgb, 
                                          dtrain=xgb.DMatrix(X_train, y_train),
                                          evals=[(xgb.DMatrix(X_val, y_val),"Valid")])
                    print('Training finito')
                    first_file = False
                    xgb_params.update({
                    'updater':'refresh',
                    'process_type': 'update',
                    'refresh_leaf': True,
                    'silent': False
                     })
                else:
                    estimator = xgb.train(xgb_params,
                                            num_boost_round=100,
                                            early_stopping_rounds=30,  
                                            feval=self.compute_rce_xgb,
                                            dtrain=xgb.DMatrix(X_train, y_train),
                                            evals=[(xgb.DMatrix(X_val, y_val),"Valid")],
                                            xgb_model = estimator)


            y_pred = estimator.predict(xgb.DMatrix(X_val))
            prauc = self.compute_prauc(y_val, y_pred)
            rce = self.compute_rce(y_val, y_pred)

            self.print_and_log('Training for {} --- PRAUC: {} / RCE: {}'.format(label, prauc, rce))
            #lgb.plot_importance(lgb_estimator, importance_type='split', max_num_features=50)
            #lgb.plot_importance(lgb_estimator, importance_type='gain', max_num_features=50)
            #plt.show()
            del df_chunk, X_train, y_train, X_val, y_val
            gc.collect()
            print('Saving model...')

            if(type_gb=='lgbm'):
                estimator.save_model('model_lgbm_{}_step_{}.txt'.format(label, n_chunk))
            elif(type_gb=='xgb'):
                pickle.dump(estimator, open('model_xgb_{}_step_{}.dat'.format(label, n_chunk), "wb"))
            n_chunk += 1

        #lgb.plot_importance(lgb_estimator, importance_type='split', max_num_features=50)
        #        lgb.plot_importance(lgb_estimator, importance_type='split', max_num_features=50)
        #lgb.plot_importance(lgb_estimator, importance_type='gain', max_num_features=50)
        #lgb.plot_importance(lgb_estimator, importance_type='gain', max_num_features=50)
        #ax = lgb.plot_tree(lgb_estimator, figsize=(15, 15), show_info=['split_gain'])
        #plt.show()
        if(type_gb=='lgbm'):
            estimator.save_model('model_lgbm_{}.txt'.format(label))
        elif(type_gb=='xgb'):
            pickle.dump(estimator, open('model_xgb_{}.dat'.format(label), "wb"))
            ax = xgb.plot_importance(estimator)
            ax.figure.savefig('importance_{}.png'.format(label))

        return estimator
    
    def gradient_boosting(self, label):
        """
            This function is used to train a gradient boosting model by means of incremental learning.
            INPUT:
                - label -> the label for the training model (Like, Retweet, Comment or Reply)
            OUTPUT:
                - trained lgbm model that will be also written on the disk
        """      
        label = label + '_engagement_timestamp'
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
        first_file = True
        not_useful_cols = ['Tweet_id', 'User_id', 'User_id_engaging', 'Reply_engagement_timestamp', 'Retweet_engagement_timestamp', 'Retweet_with_comment_engagement_timestamp', 'Like_engagement_timestamp']
        for df_chunk in pd.read_csv(self.training_file, sep='\u0001', header=None, chunksize=6000000):
            print('Processing the chunk...')
            df_chunk = self.process_chunk_tsv(df_chunk)
            df_negative = df_chunk[df_chunk[label].isna()]
            df_positive = df_chunk[~df_chunk[label].isna()]
            #n_positive = df_chunk[label].isna()
            print('Positive sample: #{} / Negative sample: #{}'.format(df_positive.shape[0], df_negative.shape[0]))
            df_negative = df_negative.sample(n=df_positive.shape[0], random_state=1)
            print('RESAMPLE -- Positive sample: #{} / Negative sample: #{}'.format(df_positive.shape[0], df_negative.shape[0]))
            print('Starting feature engineering...')
            df_chunk = pd.concat([df_positive, df_negative], axis=0, ignore_index=True)
            df_chunk = self.generate_features_lgb(df_chunk)
            df_chunk = self.encode_string_features(df_chunk)
            if(first_file):
                first_file = False
            else:
                df_training = pd.concat([df_training, df_chunk], axis=0, ignore_index=True)
            
            print('Training size: {}'.format(df_training.shape[0]))
            del df_chunk, df_negative, df_positive
            gc.collect()
            if(df_training.shape[0] > 5000000):
                print('Split training and test set')
                df_train, df_val = train_test_split(df_training, test_size=0.1)   
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

                y_pred = lgb_estimator.predict(X_val)
                prauc = self.compute_prauc(y_val, y_pred)
                rce = self.compute_rce(y_val, y_pred)

                print('Training for {} --- PRAUC: {} / RCE: {}'.format(label, prauc, rce))
                del df_training
                gc.collect()
                first_file = True
                #lgb.plot_importance(lgb_estimator, importance_type='split', max_num_features=50)
                #lgb.plot_importance(lgb_estimator, importance_type='gain', max_num_features=50)
        print('Split training and test set')
        df_train, df_val = train_test_split(df_training, test_size=0.1)   
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

        y_pred = lgb_estimator.predict(X_val)
        prauc = self.compute_prauc(y_pred, y_val)
        rce = self.compute_rce(y_pred, y_val)

        print('Training for {} --- PRAUC: {} / RCE: {}'.format(label, prauc, rce))

        lgb_estimator.save_model('model_{}.txt'.format(label))

        #lgb.plot_importance(lgb_estimator, importance_type='split', max_num_features=50)
        #        lgb.plot_importance(lgb_estimator, importance_type='split', max_num_features=50)
        #lgb.plot_importance(lgb_estimator, importance_type='gain', max_num_features=50)
        #lgb.plot_importance(lgb_estimator, importance_type='gain', max_num_features=50)
        #ax = lgb.plot_tree(lgb_estimator, figsize=(15, 15), show_info=['split_gain'])
        #plt.show()
        
        return lgb_estimator

    def save_dictionaries_on_file(self):

        os.remove('lang.json')
        f = open("lang.json","w")
        f.write(json.dumps(self.lang_dic))
        f.close()

        os.remove('tweet_type.json')
        f = open("tweet_type.json","w")
        f.write(json.dumps(self.tweet_type_dic))
        f.close()
        return

    def scalable_xgb(self, label):

        # Parameters for XGBoost
        print('Setting parameters for xgboost')

        params = {'objective': 'binary:logistic',
                'max_depth': 4, 'eta': 0.01, 'subsample': 0.5,
                'min_child_weight': 0.5}

        label = label + '_engagement_timestamp'
        not_useful_cols = ['Tweet_id', 'User_id', 'User_id_engaging', 'Reply_engagement_timestamp', 'Retweet_engagement_timestamp', 'Retweet_with_comment_engagement_timestamp', 'Like_engagement_timestamp']
        print('Import file csv')
        df_training = pd.read_csv(self.training_file, sep='\u0001', header=None, nrows=100000)
        df_training = dd.from_pandas(df_training, npartitions=3)
        #df_training = dd.read_csv(self.training_file, sep='\u0001', header=None, blocksize=10000)
        print('Read File readable')
        df_training = self.process_chunk_tsv(df_training)
        print('Starting feature engineering...')
        df_training = self.generate_features_lgb(df_training)
        df_training = self.encode_string_features(df_training, isDask=True)
        self.save_dictionaries_on_file()
        print('Prepare data for training')
        y_train = df_training[label].fillna(0)
        y_train = y_train.apply(lambda x : 0 if x == 0 else 1)
        X_train = df_training.drop(not_useful_cols, axis=1)
        print('Split training and validation')
        X_train, X_val, y_train, y_val = dask_split(X_train, y_train, test_size=0.1)
        print('Training shape: {}'.format(X_train.shape[0]))
        print('Val shape: {}'.format(X_val.shape[0]))
        print(X_train.head())
        print(y_train)
        cluster = LocalCluster()
        client = Client(cluster)

        print('Start training...')
        bst = dask_xgboost.train(client, params, X_train, y_train, num_boost_round=30)

        print('Start prediction')
        y_pred = bst.predict(client, X_val)
        prauc = self.compute_prauc(y_pred, y_val)
        rce = self.compute_rce(y_pred, y_val)

        self.print_and_log('Training for {} --- PRAUC: {} / RCE: {}'.format(label, prauc, rce))

        pickle.dump(bst, open('model_xgb_{}.dat'.format(label), "wb"))

        
        
    def generate_features_lgb(self, df):
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

    """
    ------------------------------------------------------------------------------------------
    OFFICIAL FUNCTIONS FOR EVALUATE THE SCORE
    ------------------------------------------------------------------------------------------
    """

    def compute_prauc(self, gt, pred):
        prec, recall, thresh = precision_recall_curve(gt, pred)
        prauc = auc(recall, prec)
        return prauc

    def calculate_ctr(self, gt):
        positive = len([x for x in gt if x == 1])
        ctr = positive/float(len(gt))
        return ctr

    def compute_rce(self, gt, pred):
        cross_entropy = log_loss(gt, pred)
        data_ctr = self.calculate_ctr(gt)
        strawman_cross_entropy = log_loss(gt, [data_ctr for _ in range(len(gt))])
        return (1.0 - cross_entropy/strawman_cross_entropy)*100.0

    def compute_rce_xgb(self, pred, gt):
        gt = np.asarray(gt.get_label(), dtype=np.int64)
        cross_entropy = log_loss(gt, pred, labels=[0,1])
        data_ctr = self.calculate_ctr(gt)
        strawman_cross_entropy = log_loss(gt, [data_ctr for _ in range(len(gt))], labels=[0,1])
        return 'RCE', (1.0 - cross_entropy/strawman_cross_entropy)*100.0

    def print_and_log(self, to_print):
        logging.info(to_print)
        print(to_print)
        return


    """
    ------------------------------------------------------------------------------------------
    DATASET STATISTICS
    ------------------------------------------------------------------------------------------
    """

    def count_n_lines(self):
        dd_input = dd.read_csv(self.training_file, sep='\u0001', header=None)
        n_rows = dd_input.shape[0].compute()
        self.print_and_log('The Dataframe has {} lines'.format(n_rows))
        return

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