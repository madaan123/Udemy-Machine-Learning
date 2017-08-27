# Data Preprocessing Template

# import the dataset in dataset object
dataset=read.csv('Salary_Data.csv')
#dataset= dataset[, 2:3]

"installing a new library caTools to split our datasets into 
training and testing sets ans include it by using library command
"
# install.packages('caTools')
library(caTools)
set.seed(123)
split=sample.split(dataset$Salary,SplitRatio = 2/3)
training_set=subset(dataset,split==TRUE)
test_set=subset(dataset,split==FALSE)

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

" Fitting simple linear regression to the training set 
    train the regressor on training set
"
#formula shows salary is related to years of experience
# data shows the data on which we want to perform regression 
regressor= lm(formula= Salary ~ YearsExperience,
              data=training_set)
"
we can check details of our regression by typing summary(regression on console)
it shows the number of stars higher the number more is the dependency of 
dependent column on independent set and lesser the p value more is independent
variable statistically significant
"
#Predicting our test set results
y_pred=predict(regressor, newdata = test_set)

#visualizing the results of training and test set
#install.packages('ggplot2')
library(ggplot2)
"
Now we have to plot the datapoints and linear regression line
geom_point is used to scatter plot all our obervations points of training set
aes-aesthetic function specifies the x and y variables of the plot
geom_line is used to plot the regression line
xlab and ylab for labelling x and y coordinates
"

ggplot() +
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
             colour= 'red' ) +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour='blue') +
  ggtitle('salary vs Experience(Training Set)') +
  xlab('Years of experience') +     
  ylab('salary')


ggplot() +
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary),
             colour= 'red' ) +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour='blue') +
  ggtitle('salary vs Experience(Test Set)') +
  xlab('Years of experience') +     
  ylab('salary')


