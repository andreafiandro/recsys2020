from recsysUtility import RecSysUtility

rcUtils = RecSysUtility('/datadrive/training.tsv')

rcUtils.count_action_type('Retweet_with_comment')
rcUtils.count_action_type('Retweet')
rcUtils.count_action_type('Reply')
rcUtils.count_action_type('Like')