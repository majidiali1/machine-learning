from sklearn.model_selection import  train_test_split
import matplotlib.pyplot as plt

import ml_functions as mlfn

# Load dataset 
x,y = mlfn.load_dataset(to_df=True)

# data preprocessing ?! 

# split train and test data 
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.3,random_state=42)

# plot dataset 
mlfn.dataset_scatter_plots(xtrain, ytrain) 
