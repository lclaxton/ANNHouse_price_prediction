# -*- coding: utf-8 -*-
"""
Created on Fri May 29 14:04:10 2020

House price prediction using an ANN from Kaggle

@author: Louis
"""

import pandas as pd

# Import the data set
df = pd.read_csv('kc_house_data.csv')

# Define a function to determine if the house has been renovated
def is_renovated(x):
    if x == 0:
        return 0 
    else:
        return 1 
df['is_renovated'] = df.yr_renovated.apply(is_renovated)

# Drop the unusable datatest
df.drop(['id','date','yr_renovated'],axis = 1,inplace=True)

# Determine the X and Y values  
X = df.iloc[:,1:18].values
y = df.iloc[:,0:1].values
y = y/1000



# Note don't need to do a test train spilt when using cross validation as it 
# does it for you 

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X= sc.fit_transform(X)

# importing the Keras libs

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# Defining the base model
def base_model():
    model = Sequential()
    model.add(Dense(units=20,input_dim=17,kernel_initializer = 'normal',activation = 'relu'))
    model.add(Dense(units=7))
    model.add(Dropout(0.25))
    model.add(Dense(units=1))
    model.compile(loss='mean_squared_error',optimizer='adam')
    return model

# Defining the batches and Epochs to be used and creating an object to feed 
# into the cross val score
estimator = KerasRegressor(build_fn=base_model, epochs=100, batch_size=20,verbose=1)
kfold = KFold(n_splits=10)

# Working out the scores across 10 folds of cross validation
results = cross_val_score(estimator, X, y, cv=kfold,n_jobs=4)

# The reason for doing cross validation allows you to validate how 'good' your
# model is and if it overfits
print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))
# To fit the model and predict a value need to do the below:
    
estimator.fit(X,y)
estimator.predict(X[3].reshape(1,17))


