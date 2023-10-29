import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.model_selection import  train_test_split
from sklearn.linear_model import  LinearRegression
import sklearn.metrics   as met

# Load data - Diabeties dataset 
d = datasets.load_diabetes()
X = d.data[:, np.newaxis, 2]   

# split train and test data - handi
Xtrain = X[:-20]
ytrain = d.target[:-20]

Xtest = X[-20:]
ytest = d.target[-20:]

# regression model 
model = LinearRegression()
model.fit(Xtrain , ytrain)

# model parameters
theta0 = model.intercept_ 
theta1 = model.coef_

# model prediction 
ypred = model.predict(Xtest)

# model assesment 
mse = met.mean_squared_error(ytest,ypred)

# plot results
plt.scatter(Xtest, ytest,  color='r')
plt.plot(Xtest, ypred, color='b', linewidth=2)
plt.xticks(())
plt.show()