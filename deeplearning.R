rm(list=ls(all=TRUE))
# Install packages
#https://www.youtube.com/watch?v=hd81EH1g1bE
library(keras)

# Read data
setwd("C:/Rkeras/Rkeras/")
data <- read.csv("Cardiotocographic.csv", header = T)
str(data)
dim(data)
View(data)

#######################
# Change to matrix
######################
data <- as.matrix(data)
dimnames(data) <- NULL #now does not have names

# Normalize
data[,1:21] <- normalize(data[,1:21])
data[,22] <- as.numeric(data[,22]) - 1 #values 0 1 2
summary(data)







######################
# Data partition
######################
set.seed(1234)
ind <- sample(2, nrow(data), replace = T, prob = c(0.7, 0.3))

training <- data[ind == 1, 1:21]
ncol(training)
test <- data[ind == 2, 1:21]
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
k_clear_session()
model <- keras_model_sequential()
model %>%
         layer_dense(units = 50, activation = 'relu', input_shape = ncol(training)) %>%
         layer_dense(units = 3, activation = "softmax") #softmax
summary(model)

##############################################
input * nodes + nodes
ncol(training) * 50 + 50 # layer 1
50 * 3 + 3               # layer 2
#Param layer 1 ==> 21 * 50  1 layers
#21*50 + 50 = 1100
##############################################
#Param layer 2 ==> 21 * 3
#(50 * 3) + 3 # ==> 153
##############################################

####################
# Compile
###################
model %>%
         compile(loss = "categorical_crossentropy",
                 #optimizer = "adam", # optimizer_rmsprop()
                 optimizer = optimizer_rmsprop(lr = 0.0001),
                 metrics = "accuracy")

# Fit model
history0 <- model %>%
         fit(training,
             trainLabels,
             epoch = 200,
             batch_size = 32,
             validation_split = 0.2)

history1 <- model %>%
         fit(training,
             trainLabels,
             epoch = 200,
             batch_size = 32,
             validation_split = 0.2)
history2 <- model %>%
         fit(training,
             trainLabels,
             epoch = 200,
             batch_size = 32,
             validation_split = 0.2)

history3 <- model %>%
         fit(training,
             trainLabels,
             epoch = 200,
             batch_size = 32,
             validation_split = 0.2)
plot(history0)
dev.new()
plot(history1)
dev.new()
plot(history2)
dev.new()
plot(history3)
#validation increments respect to training for loss: overfitting
 
# Evaluate model with test data
model1 <- model %>%
         evaluate(test, testLabels)

model1
#$loss
#[1] 0.3924508
#$acc
#[1] 0.8740133

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
         layer_dense(units = 50, activation = 'elu', input_shape = c(21)) %>%
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

########################
# Fine-tune model 4
########################
model4 <- keras_model_sequential()

model4 %>%
         layer_dense(units = 50, activation = 'relu', input_shape = c(21)) %>%
         layer_dense(units = 8, activation = 'relu') %>%
         layer_dense(units = 3, activation = "softmax")

summary(model4)

# Compile
model4 %>%
         compile(loss = "categorical_crossentropy",
                 optimizer = "adam",
                 metrics = "accuracy")

# Fit model
history4 <- model4 %>%
         fit(training,
             trainLabels,
             epoch = 200,
             batch_size = 32,
             validation_split = 0.2)
plot(history4)

#####################PREDICTIONS
# Prediction & confusion matrix - test data
prob4 <- model4 %>%
         predict_proba(test)

pred4 <- model4 %>%
         predict_classes(test)

table4 <- table(Predicted = pred4, Actual = testtarget)
table1
table2
table3
table4
cbind(prob, pred, testtarget)

########################
# Fine-tune model 5
########################
model5 <- keras_model_sequential()

model5 %>%
         layer_dense(units = 50, activation = 'elu', input_shape = c(21)) %>%
         layer_dense(units = 8, activation = 'elu') %>%
         layer_dense(units = 3, activation = "softmax")

summary(model5)

# Compile
model5 %>%
         compile(loss = "categorical_crossentropy",
                 optimizer = "adam",
                 metrics = "accuracy")

# Fit model
history5 <- model5 %>%
         fit(training,
             trainLabels,
             epoch = 400,
             batch_size = 32,
             validation_split = 0.2)
plot(history5)

#####################PREDICTIONS
# Prediction & confusion matrix - test data
prob5 <- model5 %>%
         predict_proba(test)

pred5 <- model5 %>%
         predict_classes(test)

table5 <- table(Predicted = pred5, Actual = testtarget)
table1
table2
table3
table4
table5
cbind(prob, pred, testtarget)

########################
# Fine-tune model 6
########################
model6 <- keras_model_sequential()

model6 %>%
         layer_dense(units = 50, activation = 'elu', input_shape = c(21)) %>%
         layer_dense(units = 8, activation = 'elu') %>%
         layer_dense(units = 3, activation = "softmax")

summary(model6)

# Compile
model6 %>%
         compile(loss = "categorical_crossentropy",
                 optimizer = optimizer_rmsprop(lr = 2e-5),
                 metrics = "accuracy")

# Fit model
history6 <- model6 %>%
         fit(training,
             trainLabels,
             epoch = 75,
             batch_size = 32,
             validation_split = 0.2)
plot(history6)

#####################PREDICTIONS
# Prediction & confusion matrix - test data
prob6 <- model6 %>%
         predict_proba(test)

pred6 <- model6 %>%
         predict_classes(test)

table6 <- table(Predicted = pred6, Actual = testtarget)
table1
table2
table3
table4
table5
cbind(prob, pred, testtarget)