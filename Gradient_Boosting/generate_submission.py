
from RecsysUtility import RecSysUtility
import pandas as pd

label_cols = ['Reply', 'Retweet_with_comment', 'Retweet', 'Like']

rsUtils = RecSysUtility('/mnt/training.tsv')

for l in label_cols:
    rsUtils.generate_submission('/mnt/val.tsv', l)

