[1mdiff --git a/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 5 - Multiple Linear Regression/multiple_linear_regression1.py b/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 5 - Multiple Linear Regression/multiple_linear_regression1.py[m
[1mindex 61b9670..001784d 100644[m
[1m--- a/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 5 - Multiple Linear Regression/multiple_linear_regression1.py[m	
[1m+++ b/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 5 - Multiple Linear Regression/multiple_linear_regression1.py[m	
[36m@@ -4,4 +4,57 @@[m [mCreated on Mon Aug 28 16:23:09 2017[m
 [m
 @author: DELL[m
 """[m
[32m+[m[32m# Data Preprocessing Template[m
 [m
[32m+[m[32m# Importing the libraries[m
[32m+[m[32mimport numpy as np # to use mathematical tools in our project[m
[32m+[m[32mimport matplotlib.pyplot as plt  # help us plot nice charts[m
[32m+[m[32mimport pandas as pd   # to import data sets and manage data sets[m
[32m+[m[32m#This dataset contains information of 50 startups[m
[32m+[m[32mdataset= pd.read_csv('50_Startups.csv')   # importing the csv file from the working directory[m
[32m+[m[32m# Creating our matrix of features[m
[32m+[m[32m# we are importing the data set of our file and skipping last column by :-1[m
[32m+[m[32mX=dataset.iloc[:, :-1].values[m[41m    [m
[32m+[m[32my=dataset.iloc[:, 4].values      # importing last column of our dataset (dependent)[m
[32m+[m
[32m+[m[32m'''[m
[32m+[m[32mNow we are going to take care of categorical data as in last column in our csv[m[41m [m
[32m+[m[32mfile.we have three types of countries and we have to encode them[m[41m [m
[32m+[m[32mas machine learning deals only with numbers and not strings[m
[32m+[m[32mwe alse have to make the labelling in such a way that there is no relational order between[m[41m [m
[32m+[m[32mdifferent encoded values(OneHotEncoder is used for that)[m
[32m+[m[32m'''[m
[32m+[m[32mfrom sklearn.preprocessing import LabelEncoder,OneHotEncoder[m
[32m+[m[32mlabelencoder_X=LabelEncoder()               # create an object of encoder class[m
[32m+[m[32mX[:,3]=labelencoder_X.fit_transform(X[:,3])  # modify the first column to the encoded values[m
[32m+[m[32monehotencoder=OneHotEncoder(categorical_features=[0])[m
[32m+[m[32mX=onehotencoder.fit_transform(X).toarray()[m
[32m+[m
[32m+[m[32m# Splitting the dataset into the Training set and Test set[m
[32m+[m[32m'''[m
[32m+[m[32mwe are going to split our datasets into training and testing sets[m
[32m+[m[32mwe are building machine learning models on datasets and then testing it on slightly[m[41m [m
[32m+[m[32mdifferent dataset to test the performance of our machine learning model[m
[32m+[m[32m'''[m
[32m+[m[32mfrom sklearn.cross_validation import train_test_split[m
[32m+[m[32mX_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.3, random_state=0)[m
[32m+[m
[32m+[m[32m# Feature Scaling[m
[32m+[m[32m'''[m
[32m+[m[32mmachine learning models work on eucledian distances and we need to scale it as it helps[m[41m [m
[32m+[m[32mthe algorithm to converge must faster[m
[32m+[m[32mNow we are going to learn how to handle scaling in machine learning that helps us to[m[41m [m
[32m+[m[32mcheck whether large data entries do not overshadow small valued entries[m
[32m+[m
[32m+[m[32mwe will recompute X_train because we want to scale it and we will transform it and just[m
[32m+[m[32mtransform test set because its already fitted to training set[m
[32m+[m[32m'''[m
[32m+[m
[32m+[m[32m"""[m
[32m+[m[32mfrom sklearn.preprocessing import StandardScaler[m
[32m+[m[32msc_X = StandardScaler()[m
[32m+[m[32mX_train = sc_X.fit_transform(X_train)[m
[32m+[m[32mX_test = sc_X.transform(X_test)[m
[32m+[m[32msc_y = StandardScaler()[m
[32m+[m[32my_train = sc_y.fit_transform(y_train)[m
[32m+[m[32m"""[m
\ No newline at end of file[m
