from recsysUtility import RecSysUtility
import pandas as pd

rsUtils = RecSysUtility('/datadrive/training.tsv')
rsUtils.generate_user_features()
rsUtils.generate_training_xgboost()
rsUtils.xgboost_training_memory('Like', training_folder='/datadrive/xgb/')
#model = rsUtils.incremental_gradient_boosting('Reply')
#model = rsUtils.incremental_gradient_boosting('Retweet')
#model = rsUtils.incremental_gradient_boosting('Retweet_with_comment')
#rsUtils.incremental_gradient_boosting('Like')
#rsUtils.incremental_gradient_boosting('Retweet')
#rsUtils.incremental_gradient_boosting('Retweet_with_comment')
#rsUtils.incremental_gradient_boosting('Reply')