from sklearn import datasets
import pandas as pd 

def load_dataset(to_df=True):
    d = datasets.load_diabetes()
    x = d.data
    y = d.target 
    
    # Transform to data frame 
    if to_df:
        x = pd.DataFrame(x) 
        y = pd.Series(y)

    return x, y 