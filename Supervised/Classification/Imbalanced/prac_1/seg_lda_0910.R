rm(list = ls())

setwd("C:/Users/Jie.Hu/Desktop/Segmentation/0910")
set.seed(1337)

#install.packages('DiscriMiner')
library(DiscriMiner)
library(tidyverse)
library(caret)


# read data
df <- read.csv('seg_0910.csv', header=T, stringsAsFactors=FALSE)


#normalize
#normalize <- function(x) {
#  return ((x - min(x)) / (max(x) - min(x)))
#}
#
#df[2:26] <- as.data.frame(lapply(df[2:26], normalize))


# Split the data into training (80%) and test set (20%)
set.seed(123)
training.samples <- df$Target %>%
  createDataPartition(p = 0.9, list = FALSE)
train.data <- df[training.samples, ]
test.data <- df[-training.samples, ]

# run da(linDA) qua(quaDA)
myda = linDA(train.data[2:26], train.data$Target)
summary(myda)
myda$functions

# predict
# classify some_data
get_classes = classify(myda, test.data[2:26])
get_classes

# compare the results against original class
table(test.data$Target, get_classes$pred_class)

#Model accuracy:
mean(test.data$Target == get_classes$pred_class)


# simulation
x <- seq(20)
acc_train <- c()
acc_test <- c()

for (i in x) {
  set.seed(i)
  training.samples <- df$Target %>%
    createDataPartition(p = 0.9, list = FALSE)
  train.data <- df[training.samples, ]
  test.data <- df[-training.samples, ]
  
  # run da
  myda = quaDA(train.data[2:26], train.data$Target)
  
  # predict
  get_classes_train = classify(myda, train.data[2:26])
  get_classes_test = classify(myda, test.data[2:26])
  
  #Model accuracy:
  acc_train[i] = (mean(train.data$Target==get_classes_train$pred_class))
  acc_test[i] = (mean(test.data$Target==get_classes_test$pred_class))
}

round(mean(acc_train), 2)
round(mean(acc_test), 2)


# crossval
# run da
myda = linDA(df[2:26], df$Target, validation="crossval")
summary(myda)
myda$functions
myda$confusion
1-myda$error_rate


# final step
# run da
myda = linDA(df[2:26], df$Target)
summary(myda)
myda$functions

# predict
get_classes = classify(myda, df[2:26])
get_classes

# compare the results against original class
table(df$Target, get_classes$pred_class)

# Model accuracy:
mean(df$Target == get_classes$pred_class)
1-myda$error_rate



