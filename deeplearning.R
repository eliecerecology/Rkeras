# Install packages
https://www.youtube.com/watch?v=hd81EH1g1bE
library(keras)
install_keras()

# Read data
data <- read.csv("Cardiotocographic.csv", header = T)
str(data)
dim(data)
# Change to matrix
data <- as.matrix(data)
dimnames(data) <- NULL #now does not have names

# Normalize
dim(data)
data[, 1:21] <- normalize(data[,1:21])
data[,22] <- as.numeric(data[,22]) - 1 #values 0 1 2
summary(data)

# Data partition
set.seed(1234)
ind <- sample(2, nrow(data), replace = T, prob = c(0.7, 0.3))
training <- data[ind==1, 1:21]
test <- data[ind==2, 1:21]

trainingtarget <- data[ind ==1, 21]
testtarget <- data[ind==2, 21]

# One Hot Encoding
trainLabels <- to_categorical(trainingtarget)
testLabels <- ---(testtarget)
print(testLabels)

# Create sequential model
model <- keras_model_sequential()
model %>%
         layer_dense(units=8, activation = ---, input_shape = ---) %>%
         ---(units = 3, activation = ---)
summary(model)

# Compile
model %>%
         compile(loss = ---,
                 optimizer = ---,
                 metrics = ---)

# Fit model
history <- model %>%
         fit(training,
             trainLabels,
             epoch = 200,
             batch_size = 32,
             validation_split = 0.2)
plot(history)

# Evaluate model with test data
model1 <- model %>%
         evaluate(test, testLabels)

# Prediction & confusion matrix - test data
prob <- model %>%
         ---(test)

pred <- model %>%
         ---(test)
table1 <- table(Predicted = pred, Actual = testtarget)

cbind(prob, pred, testtarget)

# Fine-tune model
