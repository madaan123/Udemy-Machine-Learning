# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 20:46:53 2017

@author: DELL
"""
# Data Preprocessing Template

# Importing the libraries
import numpy as np # to use mathematical tools in our project
import matplotlib.pyplot as plt  # help us plot nice charts
import pandas as pd   # to import data sets and manage data sets

dataset= pd.read_csv('Position_Salaries.csv')   # importing the csv file from the working directory
# Creating our matrix of features
# we are importing the data set of our file and skipping last column by :-1
X=dataset.iloc[:, 1:2].values    
y=dataset.iloc[:, 2].values      # importing last column of our dataset (dependent)

# Splitting the dataset into the Training set and Test set
'''
we are going to split our datasets into training and testing sets
we are building machine learning models on datasets and then testing it on slightly 
different dataset to test the performance of our machine learning model
uncomment to split the test set
'''

'''
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.3, random_state=0)
'''
# Feature Scaling
'''
machine learning models work on eucledian distances and we need to scale it as it helps 
the algorithm to converge must faster
Now we are going to learn how to handle scaling in machine learning that helps us to 
check whether large data entries do not overshadow small valued entries

we will recompute X_train because we want to scale it and we will transform it and just
transform test set because its already fitted to training set
uncomment to scale your matrix of features if required
'''

"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
"""

'''
We are going to make both linear as well as polynomial regression and compare  both of them
'''
#Fitting linear regression model to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

#Fitting Polynomial regression model to the dataset
'''
we will modify our matrix of features and it will add new independent variables that will be 
just the powers of the first independent variables
polyreg will be used to perform that task
it will also add a column of ones at the start of the transformed matrix
'''
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
X_poly=poly_reg.fit_transform(X)
lin_reg2=LinearRegression()
lin_reg2.fit(X_poly,y)

#Visualizing the linear Regression results
'''
we will plot a graph that will show the observation points and we will also plot
linear regresion line
we will plot a scatter graph that shows salaries in red colour and regression line in 
blue colour.
'''
plt.scatter(X,y,color='red')
plt.plot(X,lin_reg.predict(X),color='Blue')
plt.title('Truth or Bluff(Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

'''
Linear regression does not gives a great prediction as we are quite far from the truth
by linear prediction thats why we want a better prediction model - polynomial prediction
'''
#Visualizing the polynomial Regression results
'''
we will plot a graph that will show the observation points and we will also plot
polynomial regresion line
we will plot a scatter graph that shows salaries in red colour and regression line in 
blue colour.
'''

plt.scatter(X,y,color='red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color='Blue')
plt.title('Truth or Bluff(polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

'''
polynomial regression gives a great prediction as we are quite close to the truth
by polynomial prediction.WE can even increase the accuracy by increasing the degree of our
polynomial model
'''

'''
for more advanced plot we are using X_grid to plot all the imaginary salaries of our plot
arange provides us the range of our entries on which we have to compute our predictions
'''
X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color='red')
plt.plot(X_grid,lin_reg2.predict(poly_reg.fit_transform(X_grid)),color='Blue')
plt.title('Truth or Bluff(polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Predicting the salary using linear regression of 6.5 level
lin_reg.predict(6.5)

#Predicting the salary using polynomial regression of 6.5 level
lin_reg2.predict(poly_reg.fit_transform(6.5))
