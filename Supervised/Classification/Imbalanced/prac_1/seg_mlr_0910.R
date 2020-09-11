rm(list = ls())

setwd("C:/Users/Jie.Hu/Desktop/Segmentation/0910")
set.seed(1337)

install.packages('nnet')
library(nnet)
library(tidyverse)
library(caret)


# read data
df <- read.csv('seg_0910.csv', header=T, stringsAsFactors=FALSE)

multinomModel <- multinom(Target ~ ., data=df[2:27]) # multinom Model
summary(multinomModel) # model summary


predicted_class <- predict(multinomModel, df[2:26])

#Model accuracy:
mean(df$Target == predicted_class)



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
  multinomModel <- multinom(Target ~ ., data=train.data[2:27])
  
  # predict
  predicted_class_train <- predict(multinomModel, train.data[2:26])
  predicted_class_test <- predict(multinomModel, test.data[2:26])
  
  #Model accuracy:
  acc_train[i] = mean(train.data$Target == predicted_class_train)
  acc_test[i] = mean(test.data$Target == predicted_class_test)
}

round(mean(acc_train), 2)
round(mean(acc_test), 2)



# final step
# multinom Model
multinomModel <- multinom(Target ~ ., data=df[2:27]) 
# model summary
summary(multinomModel) 

predicted_class <- predict(multinomModel, df[2:26])

#Model accuracy:
mean(df$Target == predicted_class)