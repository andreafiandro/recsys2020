from RecsysUtility import RecSysUtility

rUtils = RecSysUtility('/mnt/training.tsv')
labels = ['Like', 'Reply', 'Retweet', 'Retweet_with_comment']

for l in labels:
    rUtils.generate_sparse_matrix('interactions', l)
    rUtils.generate_sparse_matrix('user', l)
    rUtils.generate_sparse_matrix('author', l)
    rUtils.mf_training(l)