from RecsysUtility import RecSysUtility
import pandas as pd

label_cols = ['Reply', 'Retweet_with_comment', 'Retweet', 'Like']

rsUtils = RecSysUtility('/mnt/training.tsv')

rsUtils.xgboost_multilabel()
#rsUtils.generate_submission_multilabel('/home/andreafiandro/NAS/val.tsv')
#rsUtils.generate_user_features()
#rsUtils.generate_training_xgboost(balanced=False)
#for l in label_cols:
#    rsUtils.xgboost_training_memory(l, training_folder='/mnt/xgb/')
