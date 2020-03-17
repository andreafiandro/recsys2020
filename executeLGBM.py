from recsysUtility import RecSysUtility
import pandas as pd

rsUtils = RecSysUtility('/datadrive/training.tsv')

#model = rsUtils.incremental_gradient_boosting('Reply')
#model = rsUtils.incremental_gradient_boosting('Retweet')
model = rsUtils.gradient_boosting('Retweet with comment')