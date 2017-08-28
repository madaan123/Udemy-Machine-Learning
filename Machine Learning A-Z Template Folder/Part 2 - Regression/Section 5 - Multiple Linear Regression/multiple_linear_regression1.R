#multiple linear regression

# Data Preprocessing Template

# import the dataset in dataset object
dataset=read.csv('50_Startups.csv')
#dataset= dataset[, 2:3]

"
Now we are going to take care of categorical data as in our 
last column in our csv file.we have three types of states 
in last column and we have to encode them as machine learning deals
only with numbers and not strings
factor takes three parameters first name of the column,second vector which 
contains categoris within that column and third is the vector that contains
encoded values.
"
dataset$State=factor(dataset$State,
                       levels = c('New York','California','Florida'),
                       labels = c(1,2,3))

"installing a new library caTools to split our datasets into 
training and testing sets ans include it by using library command
"
# install.packages('caTools')
library(caTools)
set.seed(123)
split=sample.split(dataset$Profit,SplitRatio = 0.8)
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

" Fitting multiple linear regression to the training set 
    train the regressor on training set
"
#formula shows Profit is related to years of experience
# data shows the data on which we want to perform regression 
regressor= lm(formula= Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
              data=training_set)

# we can also write the above statement as
# . symbolises profit depends on all independent variables
regressor= lm(formula= Profit ~ . ,data=training_set)

"
we can check details of our regression by typing summary(regression on console)
it shows the number of stars higher the number more is the dependency of 
dependent column on independent set and lesser the p value more is independent
variable statistically significant
R also handles the dummy trap and skips one dummy row
"
#Predicting our test set results
y_pred=predict(regressor, newdata = test_set)

"
Now we are going to implement backward elimination that considers only those columns or
independent variables that are statiscally more significant
we include all the columns and the remove the indignificant columns one by one
"
#Build the optimal model using backward elimination
regressor= lm(formula= Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
              data=dataset)
summary(regressor)

"
Now we want to remove those independent variables whose p values are greater than significance level
lower the p value more significant is our independent variable and we will check p values of
all the columns by calling summary function and then remove the independent variable with highest
p value and then fit the model with remaining variables and then and proceed with same steps 
until we get all p values less than significance level
"
regressor= lm(formula= Profit ~ R.D.Spend + Administration + Marketing.Spend ,
              data=dataset)
summary(regressor)

regressor= lm(formula= Profit ~ R.D.Spend + Marketing.Spend ,
              data=dataset)
summary(regressor)

regressor= lm(formula= Profit ~ R.D.Spend ,
              data=dataset)
summary(regressor)

"
Now we have got our optimal table which decides the value of our profit by backward elimination 
"
