import pandas as pd
filename = './training.tsv'
nrows = 100000
df = pd.read_csv(filename, sep='\u0001', header=None, nrows=nrows)
df.to_csv('./training_sample.tsv', sep='\u0001', header=False) 