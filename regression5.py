from sklearn.linear_model import Lasso  

from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import pandas as pd

# Load data 
X = np.array([10, 20 , 30 , 40 ,50 ,60  ,70  ,80  ,90  ,100])
y = np.array([18, 41 , 61 , 79 ,70 ,120 ,141 ,150 ,120 ,200])

X = X.reshape(-1, 1)
y = y.reshape(-1, 1)

df = pd.DataFrame(X)
df ['y'] = y

# Create LASSO model 
model = Lasso(alpha=0.1 , normalize=True)
model.fit(X , y)

# model paprameters 
feature_importance = model.coef_






