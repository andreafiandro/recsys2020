from RecsysUtility import RecSysUtility
from RecsysStats import RecSysStats
import argparse
import os

parser = argparse.ArgumentParser(description='POLINKS Solution for Recsys Challenge 2020')
parser.add_argument('--trainingpath', type=str, help='Define the path for the file training.tsv', default='./training.tsv')
parser.add_argument('--validationpath', type=str, help='Define the path for the file val.tsv', default='./val.tsv')
parser.add_argument('--testpath', type=str, help='Define the path for the file test.tsv', default='./competition_test.tsv')
parser.add_argument('--trainingfolder', type=str, help='Define the folder where the generated files will be saved', default='./')

args = parser.parse_args()

# Utilities
rsUtils = RecSysUtility(args['trainingpath'])
rStat = RecSysStats(args['trainingpath'])
label_cols = ['Reply', 'Retweet_with_comment', 'Retweet', 'Like']

# Generate the user features file

print('STEP 1: Generate user features')
rsUtils.generate_user_features(output_file=os.path.join(args['trainingfolder'], 'user_features.csv'))

print('STEP 2: Generate tweet author features')
rStat.count_author_interactions(output_file=os.path.join(args['trainingfolder'], 'author_interactions.csv'))

print('STEP 3: Generate languages features')
rsUtils.generate_spoken_languages(output_file=os.path.join(args['trainingfolder'], 'user_spoken_languages.csv'))

print('STEP 4: Generate user previous interactions')
rStat.generate_user_interaction(output_path=args['trainingfolder'])

print('STEP 5: Generate training for xgboost')
rsUtils.generate_training_xgboost(training_folder=args['trainingfolder'])

print('STEP 6: Train the four xgboost model')
for label in label_cols:
    rsUtils.xgboost_training_memory(label, training_folder=args['trainingfolder'])

print('STEP 7: Calculate prediction for the validation set')
for label in label_cols:
    rsUtils.generate_submission_speed(args['validationpath'], filename='validation')

print('Adjust submission according to CTR')
rsUtils.adjust_ctr_sub(filename='validation')

print('STEP 8: Calculate prediction for the test set')
for label in label_cols:
    rsUtils.generate_submission_speed(args['testpath'], filename='test')

print('Adjust submission according to CTR')
rsUtils.adjust_ctr_sub(filename='test')
