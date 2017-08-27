# import the dataset in dataset object
dataset=read.csv('Data.csv')

#taking care of missing data

"
is.na tells whether data is missing in the column or not 
ifelse takes three parameters one for the condition to be satisfied
one for the replacement condition and last for the else condition when condition
is false
"

dataset$Age = ifelse(is.na(dataset$Age),
                     ave(dataset$Age, FUN= function(x) mean(x,na.rm = TRUE)),
                     dataset$Age)

dataset$Salary = ifelse(is.na(dataset$Salary),
                     ave(dataset$Salary, FUN= function(x) mean(x,na.rm = TRUE)),
                     dataset$Salary)

"
Now we are going to take care of categorical data as in our first and 
last column in our csv file.we have three types of countries and 
yes/no in last column and we have to encode them as machine learning deals
only with numbers and not strings
factor takes three parameters first name of the column,second vector which 
contains categoris within that column and third is the vector that contains
encoded values.
"
dataset$Country=factor(dataset$Country,
                       levels = c('France','Spain','Germany'),
                       labels = c(1,2,3))

dataset$Purchased=factor(dataset$Purchased,
                       levels = c('Yes','No'),
                       labels = c(1,0))

"installing a new library caTools to split our datasets into 
training and testing sets ans include it by using library command
"
# install.packages('caTools')
library(caTools)
set.seed(123)
split=sample.split(dataset$Purchased,SplitRatio = 0.8)
training_set=subset(dataset,split==TRUE)
test_set=subset(dataset,split==FALSE)


