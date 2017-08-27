# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 13:22:38 2017

@author: DELL
"""
import numpy as np # to use mathematical tools in our project
import matplotlib.pyplot as plt  # help us plot nice charts
import pandas as pd   # to import data sets and manage data sets

dataset= pd.read_csv('Data.csv')   # importing the csv file from the working directory
# Creating our matrix of features
# we are importing the data set of our file and skipping last column by :-1
X=dataset.iloc[:, :-1].values    
y=dataset.iloc[:, 3].values      # importing last column of our dataset (dependent)

# taking care of missing data in our files
# a preprocessing library to preprocess datasets and Imputer takes care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy='mean',axis=0)
#now use this imputer object to fix the colums which contain missing data
X[:, 1:3] = imputer.fit_transform(X[:,1:3])  # upper bound is excluded in python


'''
Now we are going to take care of categorical data as in our first and last column in our csv 
file.we have three types of countries and yes/no in last column and we have to encode them 
as machine learning deals only with numbers and not strings
we alse have to make the labelling in such a way that there is no relational order between 
different encoded values(OneHotEncoder is used for that)
'''
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X=LabelEncoder()               # create an object of encoder class
X[:,0]=labelencoder_X.fit_transform(X[:,0])  # modify the first column to the encoded values
onehotencoder=OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()
labelencoder_Y=LabelEncoder()               # create an object of encoder class
y=labelencoder_Y.fit_transform(y)

'''
we are going to split our datasets into training and testing sets
we are building machine learning models on datasets and then testing it on slightly 
different dataset to test the performance of our machine learning model
'''
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.3, random_state=0)

'''
machine learning models work on eucledian distances and we need to scale it as it helps 
the algorithm to converge must faster
Now we are going to learn how to handle scaling in machine learning that helps us to 
check whether large data entries do not overshadow small valued entries
'''
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
'''
we will recompute X_train because we want to scale it and we will transform it and just
transform test set because its already fitted to training set
'''
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
 

