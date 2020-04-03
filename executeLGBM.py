from recsysUtility import RecSysUtility
import pandas as pd

rsUtils = RecSysUtility('//fmnas/Dataset/training.tsv')

#model = rsUtils.incremental_gradient_boosting('Reply')
#model = rsUtils.incremental_gradient_boosting('Retweet')
#model = rsUtils.incremental_gradient_boosting('Retweet_with_comment')
rsUtils.incremental_gradient_boosting('Like')
rsUtils.incremental_gradient_boosting('Retweet')
rsUtils.incremental_gradient_boosting('Retweet_with_comment')
rsUtils.incremental_gradient_boosting('Reply')