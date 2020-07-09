import pandas as pd
import sys


def remove_unnamed_column(path):
    nrows = None
    df = pd.read_csv(path, nrows=nrows)
    print('Input had:', df.columns)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    print('Output has:', df.columns)
    to_write = path.split('.csv')[0] + '_fixed.csv'
    print('Writing: %s' %to_write)
    df.to_csv(to_write, index=False, header = True)
    return


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('_______________________________________________________')
        print('This script deletes unnamed columns from a csv file')
        print('Try \"python', sys.argv[0], '<path_to_file>\"')
        print('Output will be written out to <path_to_file>_fixed')
        print('_______________________________________________________')
        exit(-1)
    path = sys.argv[1]
    print('Path: ', path)
    remove_unnamed_column(path)
    print('Finished')