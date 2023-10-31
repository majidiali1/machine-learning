import pandas as pd 

def delete_df_columns(df, columns_list):
    return df.drop(columns_list, axis=1)

def groupby(df, columns_list, method='mean'):
    return df.groupby(columns_list).mean()

