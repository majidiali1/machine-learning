from sklearn.linear_model import LinearRegression
import numpy as np 

x = np.array([1,2,3]).reshape(-1,1)
y = np.array([3,5,7]).reshape(-1,1)

# regression model 
model = LinearRegression()
model.fit(x, y) 

# model parameters
theta0 = model.intercept_ 
theta1 = model.coef_

# model prediction 
xtest  = np.array([[0], [2],[4]]).reshape(-1,1)
ypred = model.predict(xtest)

