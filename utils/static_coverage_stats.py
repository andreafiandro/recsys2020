import gc
import os
import sys
import io
import pandas as pd
from collections import Counter, defaultdict
import numpy as np

"""
col_names_training = ['Text_tokens', 'Hashtags', 'Tweet_id', 'Present_media', 'Present_links', 'Present_domains', 'Tweet_type', 'Language', 'Timestamp',
        'User_id', 'Follower_count', 'Following_count', 'Is_verified', 'Account_creation_time',
        'User_id_engaging', 'Follower_count_engaging', 'Following_count_engaging', 'Is_verified_engaging', 'Account_creation_time_engaging',
        'Engagee_follows_engager', 'Reply_engagement_timestamp', 'Retweet_engagement_timestamp', 'Retweet_with_comment_engagement_timestamp', 'Like_engagement_timestamp']

col_names_val = ['Text_tokens', 'Hashtags', 'Tweet_id', 'Present_media', 'Present_links', 'Present_domains', 'Tweet_type', 'Language', 'Timestamp',
        'User_id', 'Follower_count', 'Following_count', 'Is_verified', 'Account_creation_time',
        'User_id_engaging', 'Follower_count_engaging', 'Following_count_engaging', 'Is_verified_engaging', 'Account_creation_time_engaging',
        'Engagee_follows_engager']

def clean_list(x):
    tmp = []
    for elem in x:
        for s in elem.split('\t'):
            tmp.append(s)
    return tmp

def explode(df, lst_cols, fill_value='', preserve_index=False):
    # make sure `lst_cols` is list-alike
    if (lst_cols is not None
        and len(lst_cols) > 0
        and not isinstance(lst_cols, (list, tuple, np.ndarray, pd.Series))):
        lst_cols = [lst_cols]
    # all columns except `lst_cols`
    idx_cols = df.columns.difference(lst_cols)
    # calculate lengths of lists
    lens = df[lst_cols[0]].str.len()
    # preserve original index values    
    idx = np.repeat(df.index.values, lens)
    # create "exploded" DF
    res = (pd.DataFrame({
                col:np.repeat(df[col].values, lens)
                for col in idx_cols},
                index=idx)
             .assign(**{col:np.concatenate(df.loc[lens>0, col].values)
                            for col in lst_cols}))
    # append those rows that have empty lists
    if (lens == 0).any():
        # at least one list in cells is empty
        res = (res.append(df.loc[lens==0, idx_cols], sort=False)
                  .fillna(fill_value))
    # revert the original index order
    res = res.sort_index()
    # reset index if requested
    if not preserve_index:        
        res = res.reset_index(drop=True)
    return res

def explode_list(df, col):
    s = df[col]
    i = np.arange(len(s)).repeat(s.str.len())
    return df.iloc[i].assign(**{col: np.concatenate(s)})
"""
def test():
    nrows = 10000
    #col_to_keep = ['Tweet_id','User_id']
    col_to_keep = ['Present_links', 'Present_domains', 'Timestamp', 'Hashtags']
    col_to_aggr = ['Present_links', 'Present_domains', 'Hashtags']
    col_to_del = list(set(col_names_val)-set(col_to_keep))
    filename = './manually_downloaded_val.tsv'
    df = pd.read_csv(filename, sep='\u0001', header=None, nrows=nrows)
    df.columns = col_names_val
    df = df.drop(col_to_del, axis=1)
    df = df.dropna(subset=col_to_aggr, how='all')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s').dt.date.astype(str)
    #Split multiple entries and then explode them
    df = df.assign(Hashtags=df.Hashtags.str.split('\t'))
    df = df.assign(Present_domains=df.Present_domains.str.split('\t'))
    df = df.assign(Present_links=df.Present_links.str.split('\t'))
    my_list = []
    for i, col in enumerate(col_to_aggr): 
        d = df[col].explode().dropna().reset_index().drop('index', axis=1)
        print(d.head())
        d = d.groupby(col).size().reset_index(name='count')
        print(d.head(), len(d.index), len(df[col].explode().dropna().drop_duplicates()))
        print(d.loc[d['count']>1])
        input()
        my_list.append(d.to_dict())
        print(my_list[i])
        input()

    print(df.head()) 
    grouped = df.groupby('Timestamp', as_index=False).agg(Counter)
    print(grouped.head())
    """
    for index, row in grouped.iterrows():
        print(index, row)
        break
    """
    #print(grouped.columns)
    #Eg access:
    #print(grouped.loc[grouped['Timestamp'] =='2020-02-13']['Hashtags'][0])
    """
    lst_col = ['Hashtags']
    x = df.assign(Hashtags=df.Hashtags.str.split('\t'))
    #x = explode(x, 'Hashtags')
    x = pd.DataFrame({
            col:np.repeat(x[col].values, x[lst_col].str.len()) \
            for col in x.columns.difference([lst_col]) \
        }).assign(**{lst_col:np.concatenate(x[lst_col].values)})[x.columns.tolist()]
    for v in x['Hashtags']:
        if v is not None:
                print(x)
                break
    print(x.head())
    """
    """
    output = defaultdict(set)
    for col in ['Present_links', 'Present_domains', 'Hashtags']:
        for val in df[col].dropna().drop_duplicates():
            for v in val.split('\t'):
                output[col].add(v)
    for col in output.keys():
        print(col)
        print(output[col])
    """
    """
    grouped = df.drop(['Present_links', 'Present_domains'], axis=1).dropna().groupby(['Timestamp'])['Hashtags'].apply(list)\
                .apply(lambda lista: clean_list(lista)).apply(Counter)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        for k, v in grouped:
            print(k)
            print v
        #print('___',grouped[1])
    print(grouped.head())
    #print(df.head())

    input()
    """


def read_dataframes(filename: str, per_day=False) -> dict:
    nrows = None #Set for debug

    cols = ['Hashtags', 'Present_links', 'Present_domains', 'Timestamp'] #cols 1,4,5,8 in order
    col_ids = [1, 4, 5, 8]
    col_to_agg = ['Hashtags', 'Present_links', 'Present_domains']
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    col_time = 'Timestamp'
    col_count = 'count'
    output_dataframes = {}

    if per_day == False: #remove timestamp if not used
        col_ids.pop()
        cols.pop()

    df = pd.read_csv(filename, sep='\u0001', header=None, nrows=nrows, usecols=col_ids) #use_cols = cols ids
    df.columns = cols
    df = df.dropna(subset=col_to_agg, how='all')
    #Split multiple entries and then explode them
    df = df.assign(Hashtags=df.Hashtags.str.split('\t'))
    df = df.assign(Present_domains=df.Present_domains.str.split('\t'))
    df = df.assign(Present_links=df.Present_links.str.split('\t'))

    if per_day == False:
        for i, col in enumerate(col_to_agg): 
            d = df[col].explode().dropna().reset_index().drop('index', axis=1)
            output_dataframes[col+'_'+col_count] = len(d.index)
            #print(d.head())
            d = d.groupby(col).size().reset_index(name=col_count)
            #print(d.head())
            output_dataframes[col] = d
            #List of dataframe (col, occurrences)
            #print(d.loc[d['count']>1])
    else:
        df[col_time] = pd.to_datetime(df[col_time], unit='s').dt.day_name().astype(str)
        for col in col_to_agg: 
            d = df.drop(df.columns.difference([col_time, col]), axis=1).dropna(how='any')
            #print(d.head(), len(d.index))
            d = d.explode(col).reset_index().drop('index', axis=1).dropna()
            #print(d.head(), len(d.index))
            output_dataframes[col+'_'+col_count] = len(d.index) #total occurrences of col 
            #print(d.head())
            d = d.groupby([col_time, col]).size().reset_index(name=col_count)
            #print(d.head())
            output_dataframes[col] = d # day, col, occurrences
            df_per_day = d.drop(col, axis=1).groupby(col_time, as_index=False).agg(sum)
            #print(df_per_day, df_per_day['count'].sum())
            for day in days_of_week:
                #print(day)
                #print(df_per_day.loc[df_per_day[col_time] == day].values[0][1])
                output_dataframes[col+'_'+day] = df_per_day.loc[df_per_day[col_time] == day].values[0][1] #Total occurrences per day
                
            
        #grouped = df.groupby('Timestamp', as_index=False).agg(Counter)
        #print(grouped.head())
    
    return output_dataframes

    
def create_statistics(dfs_a, dfs_b, output, per_day = False) -> None:
    cols = ['Hashtags', 'Present_links', 'Present_domains']
    col_count = 'count'
    col_time = 'Timestamp'
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    if per_day == False:
        for col in cols:
            d_a = dfs_a[col]
            d_b = dfs_b[col]
            unique_a = d_a[col].drop_duplicates()
            unique_b = d_b[col].drop_duplicates()
            output.write('##### '+ col + ' #####\n')
            output.write('File A %s total occurrences: %d for %d unique keys\n' %(col, dfs_a[col+'_'+col_count], len(unique_a)))
            output.write('File B %s total occurrences: %d for %d unique keys\n' %(col, dfs_b[col+'_'+col_count], len(unique_b)))
            d_intersect = pd.merge(d_b, d_a, how='inner', on=col)[col]
            output.write('A intersection B is len: %d\n' %len(d_intersect.index))
            #print(d_intersect.head(), len(d_intersect.index))
            #print(pd.merge(d_intersect, d_a, how='inner', on=col).head())
            d_intersect_occ_a = pd.merge(d_intersect, d_a, how='inner', on=col)[col_count].sum()
            output.write('Sum of A intersection B keys occurrences in A / total_occuccences_of_A  = %d / %d = %f\n' \
                    %(d_intersect_occ_a, dfs_a[col+'_'+col_count], d_intersect_occ_a/dfs_a[col+'_'+col_count])    )

            d_intersect_occ_b = pd.merge(d_intersect, d_b, how='inner', on=col)[col_count].sum()
            output.write('Sum of A intersection B keys occurrences in B / total_occuccences_of_B  = %d / %d = %f\n' \
                    %(d_intersect_occ_b, dfs_b[col+'_'+col_count], d_intersect_occ_b/dfs_b[col+'_'+col_count])    )
            output.write('\n')
    else:
        for col in cols:
            d_a = dfs_a[col]
            d_b = dfs_b[col]
            unique_a = d_a[col].drop_duplicates()
            unique_b = d_b[col].drop_duplicates()
            output.write('##### '+ col + ' #####\n')
            output.write('File A %s total per day occurrences: %d for %d unique (%s) keys\n' %(col, dfs_a[col+'_'+col_count], len(unique_a), col))
            output.write('File B %s total per day occurrences: %d for %d unique (%s) keys\n' %(col, dfs_b[col+'_'+col_count], len(unique_b), col))

            d_intersect = pd.merge(unique_a, unique_b, how='inner', on=col)
            #d_intersect.drop(d_intersect.columns.difference([col_time, col]), axis=1, inplace=True)
            output.write('len(AiB) = len(A intersection B) = %d unique %s\n' %(len(d_intersect.index), col))

            intersect_occ_a = pd.merge(d_intersect, d_a, how='inner', on=col)[col_count].sum()
            intersect_occ_b = pd.merge(d_intersect, d_b, how='inner', on=col)[col_count].sum()
            output.write('Sum of A intersection B keys occurrences in A / total_occuccences_of_A  = %d / %d = %f\n' \
                    %(intersect_occ_a, dfs_a[col+'_'+col_count], intersect_occ_a/dfs_a[col+'_'+col_count])    )
            output.write('Sum of A intersection B keys occurrences in B / total_occuccences_of_B  = %d / %d = %f\n' \
                    %(intersect_occ_b, dfs_b[col+'_'+col_count], intersect_occ_b/dfs_b[col+'_'+col_count])    )

            output.write('Occ per day  \t File A \t File B \t\t (AiB_keys*occ_A_of_day)/A \t\t (AiB_keys*occ_B_of_day)/B\n')
            for day in days_of_week:
                intersect_occ_day_a = pd.merge(d_intersect, d_a.loc[d_a[col_time] == day], how='inner', on=col)[col_count].sum()
                intersect_occ_day_b = pd.merge(d_intersect, d_b.loc[d_b[col_time] == day], how='inner', on=col)[col_count].sum()
                output.write('%s: \t\t %d \t\t %d \t\t\t %f \t\t\t\t\t\t %f\n' \
                    %(day, dfs_a[col+'_'+day], dfs_b[col+'_'+day], intersect_occ_day_a/ dfs_a[col+'_'+day], intersect_occ_day_b/dfs_b[col+'_'+day]))
            output.write('\n')


def process_files(argv: list, output, per_day=False) -> None:
    dataframes_a = dataframes_b = dataframes_next = None
    for i in range(len(argv)-1):

        if os.path.isfile(argv[i]) == False:
            print('File %s do not exists' %argv[i])
            output.write('File %s do not exists' %argv[i])
            output.write('__________________________________\n')
            continue
        
        #dataframes_a = read_dataframes(argv[i]) if i == 0 else dataframes_next
        dataframes_a = read_dataframes(argv[i], per_day=per_day) if i == 0 else dataframes_next

        for j in range(i+1, len(argv)):
            if os.path.isfile(argv[j]) == False:
                print('File %s do not exists' %argv[j])
                output.write('__________________________________\n')
                continue
            #dataframes_b = read_dataframes(argv[j])
            dataframes_b = read_dataframes(argv[j], per_day=per_day)
            dataframes_next = dataframes_b if j == i+1 else dataframes_next
            output.write('Evaluating file A= %s \n\twith file B= %s\n\n' %(argv[i], argv[j]))
            #create_statistics(dataframes_a, dataframes_b, output)
            create_statistics(dataframes_a, dataframes_b, output, per_day=per_day)
            output.write('___________________________________\n\n')


def main(argv: list) -> None:
    print('Number of arguments:', len(argv), 'arguments.')
    print('Argument List:', str(argv))
    if len(argv) < 2:
        print('Need at minimumt two files.')
        return
    output_file = './output_statistics.txt'
    output = io.open(output_file, 'w', encoding='utf-8')
    output.write('_________________________________________________________________________________________________________\n\n')
    process_files(argv, output)
    output.write('_________________________________________________________________________________________________________\n\n')
    process_files(argv, output, True)
    output.write('_________________________________________________________________________________________________________\n')
    output.close()


if __name__ == "__main__":
    main(sys.argv[1:])
    
