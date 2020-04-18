from recsysUtility import RecSysUtility

rcUtils = RecSysUtility('/datadrive/val.tsv')

#rcUtils.count_n_lines()
rcUtils.count_n_tweets(isVal=True)
rcUtils.user_or_author(isVal=True)
rcUtils.count_action_type('Retweet_with_comment', isVal=True)
rcUtils.count_action_type('Retweet', isVal=True)
rcUtils.count_action_type('Reply', isVal=True)
rcUtils.count_action_type('Like', isVal=True)
