from RecsysStats import RecSysStats
from RecsysUtility import RecSysUtility

rStat = RecSysStats('/mnt/training.tsv')
rUtils = RecSysUtility('/mnt/training.tsv')

labels = ['Like', 'Reply', 'Retweet', 'Retweet_with_comment']

#n_lines = rStat.count_n_lines()
n_lines = 124991831

for l in labels:
    rUtils.generate_mf_csv(l, n_lines-4000000)