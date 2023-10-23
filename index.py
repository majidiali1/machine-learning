import pandas as pd
import numpy as np

## create a data frame 
df = pd.DataFrame([ [np.nan, 2, np.nan, 0], 
                    [3, 4, np.nan, 1], 
                    [np.nan, np.nan, np.nan, 5], 
                    [np.nan, 3, np.nan, 4],
                    [np.nan, np.nan, np.nan, np.nan] ],
                    columns=list('ABCD'))


###  Data Cleaning 
## Numnber of missing values  
nMiss = df.isnull().sum()

## fill missing values 
df1 = df.fillna(0) #fill with a specific number 
df2 = df.fillna(value={'A': 0, 'B': 1, 'C': 2, 'D': 3})  #fill with a specific number for each feature
df3 = df.fillna(df.mean()) #fill with a mean of column 
df4 = df.fillna(method='ffill') #fill with the previous record
df5 = df.dropna() #drop all NAs
df6 = df.dropna(axis=0)  #drop NAs for featurs
dfa = df['A'].dropna() #drop NAs of a featurs
df7 = df.dropna(thresh=2)  #drop rows that have not at least 2 non-NaN values
df8 = df.dropna(how='all')  #only drop rows where all columns are NaN

## missing value imputation 
from sklearn.preprocessing import Imputer
imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
imr = imr.fit(df)
imputed_data = imr.transform(df.values)

### Data Transformation 
## min max scaler
from sklearn.preprocessing import MinMaxScaler
data = [[1, 50], [2, 30],[3,40]]
scaler = MinMaxScaler()
scaler.fit(data)
scaler.transform(data)

## data normalization 
from sklearn.preprocessing import normalize
n1=normalize(data,norm='l1',axis=0) 
n2=normalize(data,norm='l2',axis=0)