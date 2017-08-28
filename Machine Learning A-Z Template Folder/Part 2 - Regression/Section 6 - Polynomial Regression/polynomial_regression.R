# Data Preprocessing Template

# import the dataset in dataset object
dataset=read.csv('Position_Salaries.csv')
dataset= dataset[2:3]

"installing a new library caTools to split our datasets into 
training and testing sets ans include it by using library command
uncomment to split the dataset into test and training set
"
"
# install.packages('caTools')
library(caTools)
set.seed(123)
split=sample.split(dataset$Purchased,SplitRatio = 0.8)
training_set=subset(dataset,split==TRUE)
test_set=subset(dataset,split==FALSE)
"

"
machine learning models work on eucledian distances and we need to scale it as 
it helps the algorithm to converge must faster
Now we are going to learn how to handle scaling in machine learning that 
helps us to check whether large data entries do not overshadow 
small valued entries
we have to include only those columns for scaling who have numeric values in dataset
"
# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

"
We are going to make both linear as well as polynomial regression and compare 
both of them
"

#Fitting linear regression model to the dataset
lin_reg = lm(Salary ~ . ,
             data = dataset)

"
we will modify our matrix will add new independent variables that will be just the 
powers of the first independent variables poly_reg will be used to perform that task
"

#Fitting polynomial regression model to the dataset
dataset$level2= dataset$Level^2    # create a new level that is square of level 1
dataset$level3= dataset$Level^3
dataset$level4= dataset$Level^4
poly_reg = lm(Salary ~ . ,
              data = dataset)
summary(poly_reg)

#Visualizing the linear Regression results
"
we will plot a graph that will show the observation points and we will also plot
linear regresion line
we will plot a scatter graph that shows salaries in red colour and regression line in 
blue colour.
"
library(ggplot2)
