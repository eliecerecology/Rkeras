rm(list=ls(all=TRUE))
library(keras)
#mnist <- dataset_mnist()
#x_train <- mnist$train$x
#y_train <- mnist$train$y
#x_test <- mnist$test$x
#y_test <- mnist$test$y

mnist <- dataset_mnist()
class(mnist)
c(c(train_images, train_labels), c(test_images, test_labels)) %<-% mnist

dim(train_images)


train_images <- array_reshape(train_images, c(60000, 28, 28, 1))
train_images <- train_images / 255
test_images <- array_reshape(test_images, c(10000, 28, 28, 1))
test_images <- test_images / 255

#one hot encoding
train_labels <- to_categorical(train_labels)
test_labels <- to_categorical(test_labels)



model <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
                input_shape = c(28, 28, 1)) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu")

  summary(model)

model <- model %>% 
  layer_flatten() %>% 
  layer_dense(units = 64, activation = "relu") %>% 
  layer_dense(units = 10, activation = "softmax")