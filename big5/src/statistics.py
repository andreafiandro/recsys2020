import pandas as pd
from big5RecsysUtility import Big5Recsys
from mlxtend.preprocessing import minmax_scaling
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr


filename = "../input/training_chunk_140.csv"
# df = pd.read_csv("../input/training_chunk_140.csv")
# print(df.head(), df.shape)
# exit()

b5u = Big5Recsys()
dftl = b5u.map_output_features(filename)
df = dftl.drop(columns=['Text_tokens'])

b5df = pd.read_csv("../output/big5_chunk_140.csv", header=None)
b5df.columns = ['O', 'C', 'E', 'A', 'N']
# print(b5df.head(), b5df.shape)
b5df = minmax_scaling(b5df, columns=['O', 'C', 'E', 'A', 'N'])
df = df.join(b5df)
df.columns = ['reply', 'retweet', 'rwc', 'like', 'O', 'C', 'E', 'A', 'N']
print(df.head(), df.shape)

f = plt.figure(figsize=(19, 15))
plt.matshow(df.corr(), fignum=f.number)
plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=45)
plt.yticks(range(df.shape[1]), df.columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16)
plt.savefig("../img/corrmatrix.png")

print(df.count())


def count_one(df, column):
    seriesObj = df.apply(lambda x: True if x[column] == 1 else False, axis=1)
    numOfRows = len(seriesObj[seriesObj == True].index)
    print('Number of Rows in dataframe in which ' + column + ': ', numOfRows)


for column in ['reply', 'retweet', 'rwc', 'like']:
    count_one(df, column)

for column in ['reply', 'retweet', 'rwc', 'like']:
    for big5 in ['O', 'C', 'E', 'A', 'N']:
        spcorr, _ = spearmanr(df[column], df[big5])
        print(column + '/' + big5 + ' Spearman\'s correlation: %.3f' % spcorr)
        peacorr, _ = pearsonr(df[column], df[big5])
        print(column + '/' + big5 + ' Pearson\'s correlation: %.3f' % peacorr)
