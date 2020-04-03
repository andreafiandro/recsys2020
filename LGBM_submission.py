
from recsysUtility import RecSysUtility
import pandas as pd

label_cols = ['Reply', 'Comment', 'Retweet', 'Like']

rsUtils = RecSysUtility('//fmnas/Dataset/training.tsv')
#rsUtils.generate_submission('/home/andreafiandro/NAS/val.tsv', 'Comment')
rsUtils.generate_user_features()