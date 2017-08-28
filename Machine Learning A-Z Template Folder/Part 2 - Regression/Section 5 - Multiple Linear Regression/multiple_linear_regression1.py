# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 16:23:09 2017

@author: DELL
"""
# Data Preprocessing Template

# Importing the libraries
import numpy as np # to use mathematical tools in our project
import matplotlib.pyplot as plt  # help us plot nice charts
import pandas as pd   # to import data sets and manage data sets
#This dataset contains information of 50 startups
dataset= pd.read_csv('50_Startups.csv')   # importing the csv file from the working directory
# Creating our matrix of features
# we are importing the data set of our file and skipping last column by :-1
X=dataset.iloc[:, :-1].values    
y=dataset.iloc[:, 4].values      # importing last column of our dataset (dependent)

'''
Now we are going to take care of categorical data as in last column in our csv 
file.we have three types of countries and we have to encode them 
as machine learning deals only with numbers and not strings
we alse have to make the labelling in such a way that there is no relational order between 
different encoded values(OneHotEncoder is used for that)
'''
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X=LabelEncoder()               # create an object of encoder class
X[:,3]=labelencoder_X.fit_transform(X[:,3])  # modify the first column to the encoded values
onehotencoder=OneHotEncoder(categorical_features=[3])
X=onehotencoder.fit_transform(X).toarray()

# Avoiding the dummy variable trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
'''
we are going to split our datasets into training and testing sets
we are building machine learning models on datasets and then testing it on slightly 
different dataset to test the performance of our machine learning model
'''
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.2, random_state=0)

# Feature Scaling
'''
machine learning models work on eucledian distances and we need to scale it as it helps 
the algorithm to converge must faster
Now we are going to learn how to handle scaling in machine learning that helps us to 
check whether large data entries do not overshadow small valued entries

we will recompute X_train because we want to scale it and we will transform it and just
transform test set because its already fitted to training set
'''

"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
"""

#Fitting multiple linear regression to the training set
''' Fitting multiple linear regression to the training set 
    train the regressor on training set
'''
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()        # create object of linearRegressor class
regressor.fit(X_train,y_train)

# predicting the test set results
y_pred=regressor.predict(X_test)


 




