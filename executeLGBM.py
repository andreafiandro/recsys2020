from recsysUtility import RecSysUtility
import pandas as pd

rsUtils = RecSysUtility('/home/andreafiandro/NAS/training.tsv')

#model = rsUtils.incremental_gradient_boosting('Reply')
#model = rsUtils.incremental_gradient_boosting('Retweet')
model = rsUtils.incremental_gradient_boosting('Retweet with comment')