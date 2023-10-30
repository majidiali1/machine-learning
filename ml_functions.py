from sklearn import datasets 
import matplotlib.pyplot as plt

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

def dataset_scatter_plots(xtrain, ytrain):
    num_cols = len(xtrain.columns)

    # Creating multiple figures with 2x2 subplots in each
    for f in range((num_cols + 3) // 4):
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
        axes = axes.flatten()
        for i in range(4 * f, min(4 * (f + 1), num_cols)):
            ax = axes[i - 4 * f]
            row = i % 4 // 2
            col = i % 2
            ax.scatter(xtrain[xtrain.columns[i]], ytrain, color='b', alpha=0.5)
            #ax.set_title(f'{xtrain.columns[i]} vs ytrain')
            ax.set_xlabel(xtrain.columns[i])
            ax.set_ylabel('ytrain')
        plt.tight_layout()
        plt.show()