# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 20:46:53 2017

@author: DELL
"""
# Regression Template

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

we will have to apply feature scaling in the SVR model as it does not supports feature 
scaling on its own
'''


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
sc_y = StandardScaler()
y = sc_y.fit_transform(y)


#Fitting SVR model to the dataset
#Create your regressor here
# importing the SVR library from sklearn and creating its object regressor
from sklearn.svm import SVR
regressor=SVR(kernel = 'rbf')
regressor.fit(X,y)

#Predicting the salary using regression of 6.5 level
# the value 6.5 is not scaled according to our previous scaling that is done on X
# we will have to use the scaler and transform the value 6.5 for predicting the results on it
# Transform method expects a array as its input so we transform 6.5 into array by using np.aaray Fx:- 
# The predicted values of the model have to be inverted as we have scaled it earlier 
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))


#Visualizing the Regression results
'''
we will plot a graph that will show the observation points and we will also plot
polynomial regresion line
we will plot a scatter graph that shows salaries in red colour and regression line in 
blue colour.
'''

plt.scatter(X,y,color='red')
plt.plot(X, regressor.predict((X)),color='Blue')
plt.title('Truth or Bluff(SVR Regression Model)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

'''
for more advanced plot we are using X_grid to plot all the imaginary salaries of our plot
arange provides us the range of our entries on which we have to compute our predictions
for higher resolution and smoother curve
'''
X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict((X_grid)),color='Blue')
plt.title('Truth or Bluff(SVR Regression Model)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

