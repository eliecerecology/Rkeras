rm(list=ls(all=TRUE))
# Install packages
#https://www.youtube.com/watch?v=hd81EH1g1bE
library(keras)

# Read data
setwd("C:/Rkeras/Rkeras/")
data <- read.csv("Cardiotocographic.csv", header = T)
str(data)
dim(data)
# Change to matrix
data <- as.matrix(data)
dimnames(data) <- NULL #now does not have names

# Normalize

data[,1:21] <- normalize(data[,1:21])
data[,22] <- as.numeric(data[,22]) - 1 #values 0 1 2
summary(data)

# Data partition
set.seed(1234)
ind <- sample(2, nrow(data), replace = T, prob = c(0.7, 0.3))
training <- data[ind==1, 1:21]
test <- data[ind==2, 1:21]
dim(training)
(1523/100)*80
##########FINAL forrmat
trainingtarget <- data[ind == 1, 22]
testtarget <- data[ind == 2, 22]

# One Hot Encoding
trainLabels <- to_categorical(trainingtarget)
testLabels <- to_categorical(testtarget)
print(testLabels)
print(trainLabels)

# Create sequential model
model <- keras_model_sequential()
model %>%
         layer_dense(units = 50, activation = 'relu', input_shape = c(21)) %>%
         layer_dense(units = 3, activation = "sigmoid")
summary(model)

####################
# Compile
###################

model %>%
         compile(loss = "categorical_crossentropy",
                 optimizer = "adam",
                 metrics = "accuracy")

# Fit model
history <- model %>%
         fit(training,
             trainLabels,
             epoch = 200,
             batch_size = 32,
             validation_split = 0.2)
plot(history)
#validation increments respect to training for loss: overfitting
 
# Evaluate model with test data
model1 <- model %>%
         evaluate(test, testLabels)

model1
#$loss
#[1] 0.4124508
#$acc
#[1] 0.8640133

#####################PREDICTIONS
# Prediction & confusion matrix - test data
prob <- model %>%
         predict_proba(test)

pred <- model %>%
         predict_classes(test)

table1 <- table(Predicted = pred, Actual = testtarget)

cbind(prob, pred, testtarget)

#######################
# Fine-tune model 2
#######################
model2 <- keras_model_sequential()
model2 %>%
         layer_dense(units = 50, activation = 'relu', input_shape = c(21)) %>%
         layer_dense(units = 3, activation = "sigmoid")

summary(model2)

# Compile
model2 %>%
         compile(loss = "categorical_crossentropy",
                 optimizer = "adam",
                 metrics = "accuracy")

# Fit model
history2 <- model2 %>%
         fit(training,
             trainLabels,
             epoch = 200,
             batch_size = 32,
             validation_split = 0.2)
plot(history2)

#####################PREDICTIONS
# Prediction & confusion matrix - test data
prob2 <- model2 %>%
         predict_proba(test)

pred2 <- model2 %>%
         predict_classes(test)

table2 <- table(Predicted = pred2, Actual = testtarget)
table1
cbind(prob, pred, testtarget)

########################
# Fine-tune model 3
########################
model3 <- keras_model_sequential()

model3 %>%
         layer_dense(units = 50, activation = 'relu', input_shape = c(21)) %>%
         layer_dense(units = 8, activation = 'relu') %>%
         layer_dense(units = 3, activation = "sigmoid")

summary(model3)

# Compile
model3 %>%
         compile(loss = "categorical_crossentropy",
                 optimizer = "adam",
                 metrics = "accuracy")

# Fit model
history3 <- model3 %>%
         fit(training,
             trainLabels,
             epoch = 200,
             batch_size = 32,
             validation_split = 0.2)
plot(history3)

#####################PREDICTIONS
# Prediction & confusion matrix - test data
prob3 <- model3 %>%
         predict_proba(test)

pred3 <- model3 %>%
         predict_classes(test)

table3 <- table(Predicted = pred3, Actual = testtarget)
table1
table2
table3
cbind(prob, pred, testtarget)

