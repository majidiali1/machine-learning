from sklearn.model_selection import  train_test_split
from sklearn.linear_model import  LinearRegression
import sklearn.metrics   as met

import ml_functions as mlfn

# Load dataset 
x,y = mlfn.load_dataset(to_df=True)

# Data preprocessing ?! 

# Split train and test data 
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.3,random_state=42)

# Plot dataset 
# mlfn.dataset_scatter_plots(xtrain, ytrain) 

# Create regression model 
model = LinearRegression()
model.fit(xtrain , ytrain)

# model inputs tunning ? over fitting 

# Model parameters 
theta0 = model.intercept_ 
theta1 = model.coef_

# Model prediction 
ypred = model.predict(xtest)

# Model assesment 
MSE = met.mean_squared_error(ytest,ypred)
print(MSE)

# plot results  