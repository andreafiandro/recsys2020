import pandas as pd
import os 
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
#from dask.diagnostics import ProgressBar
import numpy as np
import gc
import time
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, log_loss
import logging
#import dask
#import xgboost as xgb
import json
import pickle
#from RecsysStats import RecSysStats
from sklearn.multiclass import OneVsRestClassifier
#from xgboost.dask import DaskXGBClassifier
#from dask.distributed import Client, LocalCluster
#import dask.dataframe as dd


class RecSysUtility:

   
    def __init__(self, training_file):
        self.training_file = training_file
        logging.basicConfig(filename='training.log',level=logging.INFO)
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

        if(os.path.exists('col_name')):
            f = open("col_name", "r")
            self.name_of_features = f.readline().split(',')   
            f.close()
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
        for val in pd.read_csv(validation_file, sep='\u0001', header=None, chunksize=8000000):
            print('Predicting chunk {}'.format(id))
            val = self.process_chunk_tsv(val, isVal=True)
            df_out = pd.DataFrame(columns = ['Tweet_id', 'User_id', 'Prediction'])
            df_out['Tweet_id'] = val['Tweet_id']
            df_out['User_id'] = val['User_id_engaging']
            print('Starting feature engineering...')
            val = self.add_user_language(val)
            val = self.add_author_features(val)
            val = self.generate_features_lgb(val)
            val = self.add_tweet_interaction(val)
            val = self.encode_string_features(val)

            val = val.drop(not_useful_cols, axis=1)

            print('Load GB model')

            if(gb_type=='lgbm'):
                model = lgb.Booster(model_file='model_{}.txt'.format(label))
                print('Start Prediction')
                df_out['Prediction'] = model.predict(val)
            elif(gb_type=='xgb'):
                model = pickle.load(open('model_xgb_{}.dat'.format(label), "rb"))

                print('Start Prediction')
                predictions = model.predict(xgb.DMatrix(val), ntree_limit=model.best_ntree_limit)
                print(predictions)
                print(len(predictions))
                print(df_out.shape)
                df_out.loc[:, 'Prediction'] = predictions
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

    def generate_submission_speed(self, validation_file, label, filename):
        """
            Function used to generate the submission file.
            Starting from the file 
        """


        val = pd.read_csv(validation_file, sep='\u0001', header=None)
        val = self.process_chunk_tsv(val, isVal=True)
        not_useful_cols = ['Tweet_id', 'User_id', 'User_id_engaging']

        print('Starting feature engineering...')

        val = self.add_user_language(val)
        val = self.add_author_features(val)
        val = self.generate_features_lgb(val)
        val = self.add_tweet_interaction(val)
        val = self.encode_string_features(val)
        print('Da predirre')
        model = pickle.load(open('model_xgb_{}.dat'.format(label), "rb"))

        print('Start Prediction')
        preds = model.predict(xgb.DMatrix(val.drop(not_useful_cols, axis=1)), ntree_limit=model.best_ntree_limit)
        print(len(preds))
        val.loc[:,'Prediction'] = preds
        val[['Tweet_id', 'User_id_engaging', 'Prediction']].to_csv('prediction_{}_{}.csv'.format(filename, label), index=False, header=None)

    def xgboost_multilabel(self, skipr):
        """
        Classificazione Multi-Label: alleno un solo modello, con più dati possibili (da gestire la scalabilità)
        Step:
            1. Carico i dati (il più possibile)
            2. Processo i dati per pulirli
            3. Genero le features aggiuntive
            4. Split tra train/val/test
            5. Alleno il modello
            6. Salvo il modello
            7. Testo il modello
        """

        # 1. Carico i dati
        df_input = pd.read_csv(self.training_file, sep='\u0001', header=None, skiprows=skipr)

        # 2. Pulisco i dati
        df_input = self.process_chunk_tsv(df_input)
        df_input = self.generate_features_lgb(df_input, user_features_file = './user_features_final.csv')
        df_input = self.encode_string_features(df_input)
        df_input = self.add_csv_features(df_input)
        #df_input = self.generate_features_mf(df_input)

        
        # 3. Split tra Train / Val / Test
        df_train, df_test = train_test_split(df_input, test_size=0.01)
        print('Split ----- Training: {} / Test: {} '.format(df_train.shape[0], df_test.shape[0]))
        x_train, y_train = self.split_label_training(df_train)
        x_test, y_test = self.split_label_training(df_test)
        print('Parte il training... Utilizzo {} feature'.format(x_test.shape[1]))
        print(x_test.columns)
        # Alleno il modello
        clf = OneVsRestClassifier(XGBClassifier(n_jobs=-1, max_depth=10, objective='binary:logistic'))
        clf.fit(x_train, y_train)

        # Salvo il modello
        pickle.dump(clf, open('model_xgb_multilabel.dat', "wb"))

        # Testo il modello
        self.evaluate_multi_label(clf, x_test, y_test)

        return

    def generate_submission_multilabel(self, validation_file):

        not_useful_cols = ['Tweet_id', 'User_id', 'User_id_engaging']
        val = pd.read_csv(validation_file, sep='\u0001', header=None)
        val = self.process_chunk_tsv(val, isVal=True)
        df_out = pd.DataFrame(columns = ['Tweet_id', 'User_id', 'Prediction'])
        df_out['Tweet_id'] = val['Tweet_id']
        df_out['User_id'] = val['User_id_engaging']
        print('Starting feature engineering...')
        val = self.generate_features_lgb(val, user_features_file = './user_features_final.csv')
        print('Validation size: {}'.format(val.shape[0]))
        val = self.encode_val_string_features(val)
        #print('Validation size: {}'.format(val.shape[0]))
        #val = self.generate_features_mf(val, isVal=True)
        print('Validation size: {}'.format(val.shape[0]))
        val = self.add_csv_features(val, csv_path='/mnt/recsys2020_submission/test_chunk_like_prediction.csv')
        print('Da predirre')
        print(val.head(10))
        val = val.drop(not_useful_cols, axis=1)
        model = pickle.load(open('model_xgb_multilabel.dat', "rb"))

        print('Start Prediction')
        predictions = model.predict_proba(val)
        print('Predizioni')
        print(predictions)
        lista_label = ['Reply', 'Retweet', 'Comment', 'Like']

        for i in range(0,4):
            df_out['Prediction'] = predictions.T[i]
            df_out.to_csv('prediction_{}.csv'.format(lista_label[i]), index=False, header=False)

        return

    def evaluate_multi_label(self, clf, x, y):
        predictions = clf.predict_proba(x)
        lista_label = y.columns.values
        for i in range(0, 4):
            gt = y.iloc[:,i]
            predictions_label = predictions.T[i]
            rce = self.compute_rce(predictions_label, gt)
            prauc = self.compute_prauc(predictions_label, gt)
            print('{} --- PRAUC {} / RCE {}'.format(lista_label[i], prauc, rce))
        return

    def xgboost_training_memory(self, label, training_folder='/mnt/xgb/'):
        """
            This function is used to train a gradient boosting model by means of incremental learning.
            INPUT:
                - label -> the label for the training model (Like, Retweet, Comment or Reply) 
            OUTPUT:
                - trained lgbm model that will be also written on the disk
        """     
        tot_lines = 123126259
        spw_like = (tot_lines - 52719548) / 52719548
        spw_reply = (tot_lines - 3108287) / 3108287
        spw_retweet = (tot_lines - 13264420) / 13264420
        spw_comment = (tot_lines - 884082) / 884082
        
        spw = {
            'Like': spw_like,
            'Reply': spw_reply,
            'Retweet': spw_retweet,
            'Retweet_with_comment': spw_comment
        }

        # Da rimuovere
        #self.name_of_features.remove('label')

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
            'eta':0.09, 
            'tree_method': 'gpu_hist',
            'sampling_method': 'gradient_based',
            'subsample': 0.1,
            'objective': 'binary:logistic',
            'seed':1,
            'max_depth': 5,
            'disable_default_eval_metric': 1,
            'max_delta_step': 5
            #'scale_pos_weight': spw[label]
        }
        training_set = xgb.DMatrix('{}training_{}.csv?format=csv&label_column=0#dtrain.cache'.format(training_folder, label), feature_names=self.name_of_features)
        #val_set = xgb.DMatrix('{}validation_{}.csv?format=csv&label_column=0#cacheprefix'.format(training_folder, label))
        val_set = xgb.DMatrix('{}validation_{}.csv?format=csv&label_column=0#dval.cache'.format(training_folder, label), feature_names=self.name_of_features)
        evallist = [(val_set, 'eval'), (training_set, 'train')]

        print('Start training for label {}...'.format(label))

        estimator = xgb.train(xgb_params,
                                num_boost_round=200,
                                early_stopping_rounds=10,
                                feval=self.compute_rce_xgb,
                                maximize=True, 
                                evals=evallist,
                                dtrain=training_set
                                )
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
        
    def evaluate_xgboost(self, validation_file):
        label_cols = ['Reply_engagement_timestamp', 'Retweet_engagement_timestamp', 'Retweet_with_comment_engagement_timestamp', 'Like_engagement_timestamp']
        not_useful_cols = ['Tweet_id', 'User_id', 'User_id_engaging']

        val = pd.read_csv(validation_file)
        df_labels = val[label_cols].fillna(0)
    
        val = self.generate_features_lgb(val)
        val = self.encode_val_string_features(val)
        val = val.drop(not_useful_cols + label_cols, axis=1)
        rce_dic = {}
        prauc_dic = {}
        for label in ['Reply', 'Retweet', 'Retweet_with_comment', 'Like']:
            model = pickle.load(open('model_xgb_{}.dat'.format(label), "rb"))
            print('Predicting {}'.format(label))
            prediction = model.predict(xgb.DMatrix(val), ntree_limit=model.best_ntree_limit)
            df_labels[label + '_engagement_timestamp'] = df_labels[label + '_engagement_timestamp'].apply(lambda x: 1 if x != 0 else 0)
            rce = self.compute_rce(prediction, df_labels[label + '_engagement_timestamp'])
            prauc = self.compute_prauc(prediction, df_labels[label + '_engagement_timestamp'])
            rce_dic[label] = rce
            prauc_dic[label] = prauc
            self.print_and_log('Label {} RCE: {} / PRAUC: {}'.format(label, rce, prauc))
        return rce_dic, prauc_dic

    """
    ------------------------------------------------------------------------------------------
    FEATURE GENERATION
    ------------------------------------------------------------------------------------------
    """
    def add_csv_features(self, df_input, feature_name = 'bert_base', csv_path='/mnt/recsys2020_submission_internal_val/internal_val_like_prediction.csv'):
        labels = ['like', 'reply', 'retweet', 'retweet_with_comment']
        for l in labels:
            print('Add features for label {}'.format(l))
            df_features = pd.read_csv(csv_path.replace('like', l))
            df_features.columns = ['Tweet_id', 'User_id_engaging', '{}_{}'.format(feature_name, l)]
            df_input = pd.merge(df_input, df_features, how='left', on=['Tweet_id', 'User_id_engaging'])
        print('Feature {} has been included'.format(feature_name))
        print(df_input.head())
    
        return df_input

    def feature_is_in_list(self, row, feature_name, list_name, only_bool=True):
        if(isinstance(row[list_name], list) == False):
            if(only_bool):
                return False
            else:
                return 0
        if(row[feature_name] in row[list_name]):
            if(only_bool):
                return True
            else:
                return row[list_name].count(row[feature_name]) -1
        else:
            if(only_bool):
                return False
            else:
                return 0

    def add_user_language(self, df, language_file='user_spoken_languages.csv'):
        print('Adding the language from {}'.format(language_file))
        df_input = pd.read_csv(language_file, converters={'Languages_spoken': eval}, error_bad_lines=False)
        print(df_input)
        df_input.columns = ['User_id_engaging', 'spoken_languages']
        print('Duplicati {}'.format(df_input.shape[0]))
        df_input.drop_duplicates(subset=['User_id_engaging'], inplace=True)
        print('No Duplicati {}'.format(df_input.shape[0]))

        print('Input file #{} lines / User Mapping {}'.format(df.shape[0], df_input.shape[0]))
        merged = pd.merge(df, df_input, how='left', on='User_id_engaging')
        print('Merged {}'.format(merged.shape))
        print(merged[['User_id_engaging', 'Language', 'spoken_languages']].head())
        merged.loc[:, 'User_spoke'] = merged.apply(lambda x: self.feature_is_in_list(x, 'Language', 'spoken_languages', only_bool=False), axis = 1)
        print(merged[['User_spoke', 'Language', 'spoken_languages']].head())
        merged = merged.drop(['spoken_languages'], axis = 1)
        return merged

    def add_tweet_interaction(self, df, interaction_file='user_interactions_Like.csv'):
        labels = ['Retweet_with_comment', 'Like', 'Reply', 'Retweet']
        for l in labels:
            print('Add features from {}'.format(interaction_file.replace('Like', l)))
            df_input = pd.read_csv(interaction_file.replace('Like', l), converters={'{}_interactions'.format(l): eval}, error_bad_lines=False)
            df_input.columns = ['User_id_engaging', '{}_interactions'.format(l)]
            print(df_input.head())
            print('Duplicati {}'.format(df_input.shape[0]))
            df_input.drop_duplicates(subset=['User_id_engaging'], inplace=True)
            print('No Duplicati {}'.format(df_input.shape[0]))

            print('Input file #{} lines / User Mapping {}'.format(df.shape[0], df_input.shape[0]))
            df = pd.merge(df, df_input, how='left', on='User_id_engaging')
            df.loc[:,'previous_{}'.format(l)] = df.apply(lambda x: self.feature_is_in_list(x, 'User_id', '{}_interactions'.format(l), only_bool=False), axis = 1)    
            print('New features for label {}'.format(l))
            print(df[df['previous_{}'.format(l)] != 0].head())
            
            df = df.drop(['{}_interactions'.format(l)], axis = 1)
        return df

    def add_hashtag_interaction(self, df, interaction_file='user_hashtags_Like.csv'):
        labels = ['Retweet_with_comment', 'Like', 'Reply', 'Retweet']
        for l in labels:
            print('Add features from {}'.format(interaction_file.replace('Like', l)))
            df_input = pd.read_csv(interaction_file.replace('Like', l))
            df_input.columns = ['User_id_engaging', 'hashtag_{}'.format(l)]
            print('Duplicati {}'.format(df_input.shape[0]))
            df_input.drop_duplicates(subset=['User_id_engaging'], inplace=True)
            print('No Duplicati {}'.format(df_input.shape[0]))

            print('Input file #{} lines / User Mapping {}'.format(df.shape[0], df_input.shape[0]))
            df = pd.merge(df, df_input, how='left', on='User_id_engaging')
            df.loc[:,'hashtags_used_{}'.format(l)] = df.apply(lambda x: self.feature_is_in_list(x, 'Hashtags', 'hashtag_{}'.format(l), only_bool=False), axis = 1)    
            print('New features for label {}'.format(l))
            print(df[df['hashtags_used_{}'.format(l)] > 1].head())
            
            df = df.drop(['interactions_{}'.format(l)], axis = 1)
        return df

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

    def generate_features_lgb(self, df, user_features_file='user_features_final.csv'):
        """
        Function to generate the features included in the gradient boosting model.
        """
        # Aggiungo le features riguardanti il testo prima che vengano perse le info

        df.loc[:,'word_retweet'] = df['Text_tokens'].apply(lambda x: self.find_word(x, '62893|12577'))
        df.loc[:,'word_share'] = df['Text_tokens'].apply(lambda x: self.find_word(x, '23867'))
        df.loc[:,'word_like'] = df['Text_tokens'].apply(lambda x: self.find_word(x, '11850'))
        df.loc[:,'word_comment'] = df['Text_tokens'].apply(lambda x: self.find_word(x, '49641'))
        df.loc[:,'word_reply'] = df['Text_tokens'].apply(lambda x: self.find_word(x, '10246|59146'))

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

        # Add time related features
        df.loc[:,'tweet_date'] = pd.to_datetime(df['Timestamp'], unit='s')
        df.loc[:,'tweet_hour'] = df['tweet_date'].dt.hour
        df.loc[:,'tweet_day'] = df['tweet_date'].dt.day
        df.loc[:,'tweet_day_week'] = df['tweet_date'].dt.dayofweek
        df.drop(['tweet_date'], axis=1, inplace=True)

        
        # Riempio i valori NaN con -1 per dare un informazione in più al gradient boosting
        col_to_fill = ['Tot_reply', 'Tot_retweet', 'Tot_comment', 'Tot_like', 'Tot_action', 'ratio_reply', 'ratio_retweet', 'ratio_comment', 'ratio_like']
        df[col_to_fill] = df[col_to_fill].fillna(value=-1)


        return df

    def add_author_features(self, df, author_features_file='author_interactions.csv'):
        df_author = pd.read_csv(author_features_file)
        df_author.columns = ['User_id', 'Tot_author_Reply' , 'Tot_author_Retweet' , 'Tot_author_Retweet_with_comment' ,
                            'Tot_author_Like', 'ntweets' , 'Reply_tweet_ratio' , 'Retweet_tweet_ratio', 'Retweet_with_comment_tweet_ratio' , 'Like_tweet_ratio']
        df = df.merge(df_author, how='left', left_on='User_id', right_on='User_id')
        return df
  
    def split_pseudo_negative(self, df):
        df_pseudo_negative = df[df['Reply_engagement_timestamp'].isna() & df['Retweet_engagement_timestamp'].isna() & df['Retweet_with_comment_engagement_timestamp'].isna() & df['Like_engagement_timestamp'].isna()]
        df_action = df[~df['Reply_engagement_timestamp'].isna() & ~df['Retweet_engagement_timestamp'].isna() & ~df['Retweet_with_comment_engagement_timestamp'].isna() & ~df['Like_engagement_timestamp'].isna()]
        return df_pseudo_negative, df_action

    """
    ------------------------------------------------------------------------------------------
        GENERAZIONE DEI FILE DI TRAINING
    ------------------------------------------------------------------------------------------
    """

    def generate_four_files(self, df_in, training_folder, file_type, balanced):
        print(df_in)
        print(df_in.columns)
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

    def generate_training_xgboost(self, training_folder='/datadrive/xgb/', val_size=0.1, test_size=0.1, training_lines=None, balanced=False, already_processed = False):

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
        if(already_processed):
            df_val = pd.read_csv(self.training_file, nrows=val_rows)
            df_val = self.process_chunk_tsv(df_val)
        else:
            df_val = pd.read_csv(self.training_file, sep='\u0001', header=None, nrows=val_rows)
            df_val = self.process_chunk_tsv(df_val)

        print('Starting feature engineering...')
        df_val = self.add_user_language(df_val)
        df_val = self.add_author_features(df_val)
        print(df_val.head())
        df_val = self.generate_features_lgb(df_val)
        df_val = self.add_tweet_interaction(df_val)
        df_val = self.encode_string_features(df_val)
        self.name_of_features = self.generate_four_files(df_val, training_folder, 'validation', balanced)
        print('Nome Features')
        print(self.name_of_features)
        self.name_of_features.remove('label')
         # Salvo i nomi delle colonne
        with open("col_name", "w") as outfile:
            outfile.write(",".join(self.name_of_features))
            
        test_rows = int(test_size * training_lines)
        print('Test Rows: {}'.format(val_rows))
        
        if(already_processed):
            df_test = pd.read_csv(self.training_file, nrows=test_rows, skiprows=val_rows)
            df_test = self.process_chunk_tsv(df_test)
        
        else:
            df_test = pd.read_csv(self.training_file, sep='\u0001', header=None, nrows=test_rows, skiprows=val_rows)
            df_test = self.process_chunk_tsv(df_test)
        print('Starting feature engineering...')
        df_test = self.add_user_language(df_test)
        df_test = self.add_author_features(df_test)
        df_test = self.generate_features_lgb(df_test)
        #df_test = self.add_hashtag_interaction(df_test)
        df_test = self.add_tweet_interaction(df_test)
        df_test = self.encode_string_features(df_test)
        

        self.generate_four_files(df_test, training_folder, 'test', balanced)

        if(already_processed):
            for df_chunk in pd.read_csv(self.training_file, chunksize=10000000, skiprows=val_rows+test_rows):

                print('Processing the chunk {}...'.format(n_chunk))
                
                n_chunk +=1
                df_chunk = self.process_chunk_tsv(df_chunk)

                print('Starting feature engineering...')
                df_chunk = self.add_user_language(df_chunk)
                df_chunk = self.add_author_features(df_chunk)
                df_chunk = self.generate_features_lgb(df_chunk)
                #df_chunk = self.add_hashtag_interaction(df_chunk)
                df_chunk = self.add_tweet_interaction(df_chunk)
                df_chunk = self.encode_string_features(df_chunk)
                print(df_chunk.head())
                self.generate_four_files(df_chunk, training_folder, 'training', balanced)
        else:
            for df_chunk in pd.read_csv(self.training_file, sep='\u0001', header=None, chunksize=10000000, skiprows=val_rows+test_rows):
                
                print('Processing the chunk {}...'.format(n_chunk))
                
                n_chunk +=1
                df_chunk = self.process_chunk_tsv(df_chunk)

                print('Starting feature engineering...')
                df_chunk = self.add_user_language(df_chunk)
                df_chunk = self.add_author_features(df_chunk)
                df_chunk = self.generate_features_lgb(df_chunk)
                #df_chunk = self.add_hashtag_interaction(df_chunk)
                df_chunk = self.add_tweet_interaction(df_chunk)
                df_chunk = self.encode_string_features(df_chunk)
                print(df_chunk.head())
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
        pred = np.asarray(pred, dtype=np.float64)
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
    def find_word(self, row, word_to_find):
        if(word_to_find in row):
            return True
        else:
            return False

    def split_label_training(self, df):
        label_cols = ['Reply_engagement_timestamp', 'Retweet_engagement_timestamp', 'Retweet_with_comment_engagement_timestamp', 'Like_engagement_timestamp']
        not_useful_cols = ['Tweet_id', 'User_id', 'User_id_engaging']
        df_labels = df[label_cols]
        df_labels = df_labels.fillna(0)
        for c in df_labels.columns:
            df_labels[c] = df_labels[c].apply(lambda x: 1 if x != 0 else 0)
        df_train = df.drop(label_cols + not_useful_cols, axis=1)
        return df_train, df_labels

    def print_and_log(self, to_print):
        logging.info(to_print)
        print(to_print)
        return

    def reduce_mem_usage(self, df):

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

    def create_chunk_csv(self, output_dir='/mnt/val', chunk_size = 12500000):

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
            print('Chunk # {} -- Analizzate #{} righe'.format(chunk_n, chunk_n*chunk_size))
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
        df_prediction_rt = pd.read_csv(submission_file)
        df_prediction_rt.columns = ['Tweet_id', 'User_id', 'Prediction']
        df_retweet_sicuri = pd.read_csv(solution_file)
        df_retweet_sicuri = df_retweet_sicuri[['Tweet_id', 'User_id_engaging']]
        df_merged = df_prediction_rt.merge(df_retweet_sicuri, how='left', left_on=['Tweet_id', 'User_id'], right_on=['Tweet_id', 'User_id_engaging'])
        df_merged['Prediction'] = np.where((df_merged['User_id']==df_merged['User_id_engaging']), 1, df_merged['Prediction'])
        df_merged = df_merged[['Tweet_id', 'User_id', 'Prediction']]
        df_merged.to_csv('prediction_Retweet_leaks.csv', header=False, index=False)
        return
        
    def magic_algorithm(self, validation_file, filename):
        tot_lines = 123126259
        ctr_like = 52719548 / tot_lines
        ctr_reply = 3108287 / tot_lines
        ctr_retweet = 13264420 / tot_lines
        ctr_comment = 884082 / tot_lines
        df_val = pd.read_csv(validation_file, sep='\u0001', header=None)
        df_val = self.process_chunk_tsv(df_val, isVal=True)
        df_val = df_val[['Tweet_id', 'User_id_engaging']]
        
        
        df_val.loc[:,'prediction_Like'] = ctr_like 
        df_val.loc[:,'prediction_Reply'] = ctr_reply 
        df_val.loc[:,'prediction_Retweet'] = ctr_retweet 
        df_val.loc[:,'prediction_Comment'] = ctr_comment 

        lista_label = ['Reply', 'Retweet', 'Comment', 'Like']
        for l in lista_label:
            df_val[['Tweet_id', 'User_id_engaging', 'prediction_{}'.format(l)]].to_csv('prediction_{}_{}.csv'.format(filename,l), header = None, index = False)
        
        return

    def adjust_ctr_sub(self, filename):
        tot_lines = 123126259
        ctr_like = 52719548 / tot_lines
        ctr_reply = 3108287 / tot_lines
        ctr_retweet = 13264420 / tot_lines
        ctr_comment = 884082 / tot_lines
        ctr = {
            'Like': ctr_like,
            'Reply': ctr_reply,
            'Retweet': ctr_retweet,
            'Retweet_with_comment': ctr_comment
        }

        #x = PrettyTable()
        #x.field_names = ['Label','CTR', 'Top_N_Pred']
        labels = ['Like', 'Retweet_with_comment', 'Retweet', 'Reply']
        for l in labels:
            df_sub = pd.read_csv('prediction_{}_{}.csv'.format(filename, l), header=None)
            print(df_sub.head())
            print('Size iniziale: {}'.format(df_sub.shape[0]))
            df_sub.columns = ['Tweet_id', 'User_Id', 'Prediction']
            top_n_pred = int(ctr[l]*df_sub.shape[0])
            print('Aggiusto il ctr per la label {}'.format(l))
            #x.add_row([l, ctr[l], top_n_pred])
            print('Prendo solo le top {}'.format(top_n_pred))
            df_sub = df_sub.sort_values(by='Prediction', ascending = False)
            df_sub.iloc[top_n_pred:,2] = ctr[l]
            print(df_sub.head())
            print('Size Finale: {}'.format(df_sub.shape[0]))
            df_sub.to_csv('pred_adj_{}.csv'.format(l), header=None, index=False)      
        #print(x)      

