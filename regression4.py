import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# Load data 
d = pd.read_csv("files/diamonds.csv", index_col = 0)

# preprocessing - data transformation - letter to index (number)
d.color = d.color.apply(list('JIHGFED').index)
d.cut = d.cut.apply(['Fair','Good','Very Good','Premium','Ideal'].index)
d.clarity = d.clarity.apply(['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF'].index)

# describe data 
desc = d.describe() 

# declare x and y - delete target column (price) 
X = d.loc[:,(d.columns != 'price')]  
Y = d.price

# split train and test data 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.30, random_state = 1)

# new xtrain and xtest - delete some columns - which columns must be omitted ? 
new_X_train = X_train.loc[:,(X_train.columns != "cut") & (X_train.columns != "color") & (X_train.columns != "clarity")]
new_X_test  = X_test.loc [:,(X_test.columns  != "cut") & (X_test.columns  != "color") & (X_test.columns  != "clarity")]

# regression model 
model = LinearRegression()
model.fit(new_X_train, Y_train)
Y_pred = model.predict(new_X_test)

# medel assesment
MAE = mean_absolute_error(Y_test, Y_pred) # mean absolute error 
RMSE = sqrt(mean_squared_error(Y_test,Y_pred))  # radical mean squared error 

