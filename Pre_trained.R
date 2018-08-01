rm(list=ls(all=TRUE))
library(keras)

####################
#Feature extraction
####################
'''Feature extraction consists of using the representations learned
by a previous network to extract interesting features from new samples.
These features are then run through a new classifier, which is trained
from scratch.'''



#I had to downloaded manually
conv_base <- application_vgg16(
  weights = "imagenet", #which net: animals etc..but is VGG16
  include_top = FALSE,   #top dense layer present: NO
  input_shape = c(150, 150, 3) #input 3 channels
)

summary(conv_base)

###########################################################################
#Extending the model you have (conv_base) by adding dense layers on top
###########################################################################
base_dir <- "/home/rstudio/cats_and_dogs_small"
train_dir <- file.path(base_dir, "train/")
validation_dir <- file.path(base_dir, "validation/")
test_dir <- file.path(base_dir, "test/")

###################################################################
#functiones
#"image_data_generator": 
#Generate batches of image data with real-time data augmentation.
#The data will be looped over (in batches).

#flow_images_from_directory
#Generates batches of data from images in a directory (with optional 
#augmented/normalized data)

#generator_next(generator)
#Use to retrieve items from generators (e.g. image_data_generator()). 
#Will return either the next item or NULL if there are no more items.

#We will extract features from these images by calling the predict method on the model.

datagen <- image_data_generator(rescale = 1/255) #extract images as arrays
#as weel their labels
batch_size <- 20

extract_features <- function(directory, sample_count) {
  
  features <- array(0, dim = c(sample_count, 4, 4, 512))  #make an array
  labels <- array(0, dim = c(sample_count))   # make another array
  
  generator <- flow_images_from_directory( #transform the photos of the
    directory = directory,
    generator = datagen,
    target_size = c(150, 150),
    batch_size = batch_size,
    class_mode = "binary"
  )
  
  i <- 0
  while(TRUE) {
    batch <- generator_next(generator)
    inputs_batch <- batch[[1]]
    labels_batch <- batch[[2]]
    #####################################################################    
    ##We will extract features from these images by calling the predict 
    #method on the model
    ##################################################################### 
    features_batch <- conv_base %>% predict(inputs_batch)
    
    index_range <- ((i * batch_size)+1):((i + 1) * batch_size)
    features[index_range,,,] <- features_batch
    labels[index_range] <- labels_batch
    
    i <- i + 1
    if (i * batch_size >= sample_count)
      # Note that because generators yield data indefinitely in a loop, 
      # you must break after every image has been seen once.
      break
  }
  
  list(
    features = features, 
    labels = labels
  )
}

train <- extract_features(train_dir, 2000)
validation <- extract_features(validation_dir, 1000)
test <- extract_features(test_dir, 1000)

class(train)
dim(train[[1]])
dim(train[[2]])
length(train)

####################################
#RESHAPEN FOR THE DENSE LAYER
####################################

reshape_features <- function(features) {
  array_reshape(features, dim = c(nrow(features), 4 * 4 * 512))
}

dim(train$features) # 2000    4    4  512

#######################
#input and outputs
#######################

train$features <- reshape_features(train$features) #NEW INPUT
validation$features <- reshape_features(validation$features) #new validation
test$features <- reshape_features(test$features)

model <- keras_model_sequential() %>% 
  layer_dense(units = 256, activation = "relu", 
              input_shape = 4 * 4 * 512) %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  optimizer = optimizer_rmsprop(lr = 2e-5),
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

history <- model %>% fit(
  train$features, train$labels,
  epochs = 30,
  batch_size = 20,
  validation_data = list(validation$features, validation$labels)
)

plot(history)
#oevrfitting anyway

################################################################
#2nd METHOD
#################################################################
rm(list=ls(all=TRUE))
library(keras)

#I had to downloaded manually
conv_base <- application_vgg16(
  weights = "imagenet", #which net: animals etc..but is VGG16
  include_top = FALSE,   #top dense layer present: NO
  input_shape = c(150, 150, 3) #input 3 channels
)

summary(conv_base)


model <- keras_model_sequential() %>% 
  conv_base %>% 
  layer_flatten() %>% 
  layer_dense(units = 256, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

summary(model)

cat("This is the number of trainable weights before freezing",
    "the conv base:", length(model$trainable_weights), "\n")

freeze_weights(conv_base) #<-----------

cat("This is the number of trainable weights after freezing",
    "the conv base:", length(model$trainable_weights), "\n")

###########################
##START TRAINING
###########################

base_dir <- "/home/rstudio/cats_and_dogs_small"
train_dir <- file.path(base_dir, "train/")
validation_dir <- file.path(base_dir, "validation/")
test_dir <- file.path(base_dir, "test/")

train_datagen = image_data_generator(
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE,
  fill_mode = "nearest"
)

test_datagen <- image_data_generator(rescale = 1/255)

train_generator <- flow_images_from_directory(
  train_dir,
  train_datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "binary"
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  test_datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "binary"
)

####################################
#
####################################
model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 2e-5),
  metrics = c("accuracy")
)

history <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = 100,
  epochs = 30,
  validation_data = validation_generator,
  validation_steps = 50
)

plot(history)

save_model_hdf5(model, "cats_and_dogs_small_3.h5")

##############
#Check test data

test_generator <- flow_images_from_directory(
  test_dir,
  test_datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "binary"
)

model %>% evaluate_generator(test_generator, steps = 50)


