rm(list=ls(all=TRUE))
library(keras)
mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

###############################
##shape (images,width,height)
###############################

dim(x_train) #60000 images, of 28 and 28
             #nrow = 60000

dim(x_test)

dim(y_train)
dim(y_test)

dim(x_train)
dim(y_train)

# reshape
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))
# rescale
x_train <- x_train / 255
x_test <- x_test / 255

# One-hot encodig the target
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

dim(y_train)



#model
model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = 'softmax') #softmax is logistic #multiclass

summary(model)

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

history <- model %>% fit(
  x_train, 
  y_train, 
  epochs = 30, batch_size = 128, 
  validation_split = 0.2
)

plot(history)
model %>% evaluate(x_test, y_test)

###PREDICT CLASSES
pred <- model %>% predict_classes(x_test)
pred - y_test
####model 2 using ELU
#model
model2 <- keras_model_sequential() 
model2 %>% 
  layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = 'softmax') #softmax is logistic #multiclass

summary(model)

model2 %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

history2 <- model2 %>% fit(
  x_train, 
  y_train, 
  epochs = 30, batch_size = 128, 
  validation_split = 0.2,
  view_metrics = FALSE
)

plot(history2)

model2 %>% evaluate(x_test, y_test)


dim(x_train)
dim(y_train)




