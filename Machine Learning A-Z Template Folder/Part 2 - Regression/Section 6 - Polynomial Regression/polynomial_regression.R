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
dataset$Level2= dataset$Level^2    # create a new level that is square of level 1
dataset$Level3= dataset$Level^3
dataset$Level4= dataset$Level^4
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
"
Now we have to plot the datapoints and linear regression line
geom_point is used to scatter plot all our obervations points of training set
aes-aesthetic function specifies the x and y variables of the plot
geom_line is used to plot the regression line
xlab and ylab for labelling x and y coordinates
"
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour= 'red' ) +
  geom_line(aes(x = dataset$Level, y = predict(lin_reg, newdata = dataset)),
            colour='blue') +
  ggtitle('Truth or Bluff(Linear Regression)') +
  xlab('Years of experience') +     
  ylab('salary')

"
Linear regression does not gives a great prediction as we are quite far from the truth
by linear prediction thats why we want a better prediction model - polynomial prediction
"
#Visualizing the polynomial Regression results
"
we will plot a graph that will show the observation points and we will also plot
polynomial regresion line
we will plot a scatter graph that shows salaries in red colour and regression line in 
blue colour.
"
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour= 'red' ) +
  geom_line(aes(x = dataset$Level, y = predict(poly_reg, newdata = dataset)),
            colour='blue') +
  ggtitle('Truth or Bluff(Polynomial Regression)') +
  xlab('Years of experience') +     
  ylab('salary')

"
polynomial regression gives a great prediction as we are quite close to the truth
by polynomial prediction.WE can even increase the accuracy by increasing the degree of our
polynomial model
"

#Predicting a new value by linear regression
y_pred= predict(lin_reg, data.frame(Level=6.5))

#Predicting a new value by polynomial regression
y_pred= predict(poly_reg, data.frame(Level=6.5,
                                    Level2=6.5^2,
                                    Level3=6.5^3,
                                    Level4=6.5^4))

