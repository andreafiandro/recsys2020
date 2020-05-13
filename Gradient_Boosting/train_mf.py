from RecsysUtility import RecSysUtility

rUtils = RecSysUtility('/mnt/training.tsv')
labels = ['Like', 'Reply', 'Retweet', 'Retweet_with_comment']

for l in labels:
    rUtils.mf_training(l)