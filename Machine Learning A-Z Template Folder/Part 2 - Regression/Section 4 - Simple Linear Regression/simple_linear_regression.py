# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 00:29:50 2017

@author: DELL
"""

# Data Preprocessing Template

# Importing the libraries
import numpy as np # to use mathematical tools in our project
import matplotlib.pyplot as plt  # help us plot nice charts
import pandas as pd   # to import data sets and manage data sets

dataset= pd.read_csv('Salary_Data.csv')   # importing the csv file from the working directory
# Creating our matrix of features
# we are importing the data set of our file and skipping last column by :-1
X=dataset.iloc[:, :-1].values    
y=dataset.iloc[:, 1].values      # importing last column of our dataset (dependent)

# Splitting the dataset into the Training set and Test set
'''
we are going to split our datasets into training and testing sets
we are building machine learning models on datasets and then testing it on slightly 
different dataset to test the performance of our machine learning model
'''
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=1/3, random_state=0)

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

''' Fitting simple linear regression to the training set 
    train the regressor on training set
'''
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()  # Creating object of linear regression class
# fit the regressor object to our training set
regressor.fit(X_train,y_train)

# predicting the test set results
y_pred=regressor.predict(X_test)

#Visualizing the training set results
'''
we will plot a graph that will show the observation points and we will also plot
linear regresion line
we will plot a scatter graph that shows salaries in red colour and regression line in 
blue colour.
'''
plt.scatter(X_train,y_train, color='red')   # plot the salaries of the employees
plt.plot(X_train,regressor.predict(X_train), color='blue')  # plot the regression line
plt.title('salary Vs Experience(Training Set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

#Visualizing the test set results
plt.scatter(X_test,y_test, color='red')   # plot the salaries of the employees
plt.plot(X_train,regressor.predict(X_train), color='blue')  # plot the regression line
plt.title('salary Vs Experience(Test Set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()


