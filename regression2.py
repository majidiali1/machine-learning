import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
import sklearn.metrics   as met

from sklearn.model_selection import  train_test_split
from sklearn.linear_model import  LinearRegression

# Load data 
X = np.array([10, 20 , 30 , 40 ,50 ,60  ,70  ,80  ,90  ,100])
y = np.array([18, 41 , 61 , 79 ,70 ,120 ,141 ,150 ,120 ,200])

X = X.reshape(-1, 1)
y = y.reshape(-1, 1)

df = pd.DataFrame(X)
df ['y'] = y

# split train and test data 
xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size=0.3,random_state=42)

# Plot data 
plt.scatter(xtrain, ytrain ,  color='b')
plt.scatter(xtest , ytest  ,  color='r')
plt.xlabel('x')
plt.ylabel('y')
#plt.show()

# regression model 
model = LinearRegression()
model.fit(xtrain , ytrain)

# model parameters
theta0 = model.intercept_ 
theta1 = model.coef_

# model prediction 
ypred = model.predict(xtest)

# model assesment 
mse = met.mean_squared_error(ytest,ypred)

# plot results  









