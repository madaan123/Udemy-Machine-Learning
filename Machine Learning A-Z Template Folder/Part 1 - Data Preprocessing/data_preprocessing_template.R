# Data Preprocessing Template

# import the dataset in dataset object
dataset=read.csv('Data.csv')
#dataset= dataset[, 2:3]

"installing a new library caTools to split our datasets into 
training and testing sets ans include it by using library command
"
# install.packages('caTools')
library(caTools)
set.seed(123)
split=sample.split(dataset$Purchased,SplitRatio = 0.8)
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