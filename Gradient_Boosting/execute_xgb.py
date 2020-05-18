from RecsysUtility import RecSysUtility
from RecsysStats import RecSysStats

import pandas as pd

label_cols = ['Reply', 'Retweet_with_comment', 'Retweet', 'Like']

rsUtils = RecSysUtility('/mnt/training.tsv')
rStat = RecSysStats('/mnt/training.tsv')
#n_lines = rStat.count_n_lines()
n_lines = 123126259
#rsUtils.xgboost_multilabel(skipr=n_lines-4000000)
rsUtils.generate_submission_multilabel('/mnt/val.tsv')
#rsUtils.generate_user_features()
#rsUtils.generate_training_xgboost(balanced=False)
#for l in label_cols:
#    rsUtils.xgboost_training_memory(l, training_folder='/mnt/xgb/')
