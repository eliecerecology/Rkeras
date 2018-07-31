'''dir.create("~/jena_climate", recursive = TRUE)
download.file(
  "https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip",
  "~/jena_climate/jena_climate_2009_2016.csv.zip"
)
unzip(
  "~/jena_climate/jena_climate_2009_2016.csv.zip",
  exdir = "~/jena_climate"
)'''

rm(list=ls(all=TRUE))
library(tibble)
library(readr)

data_dir <- "~/jena_climate"
fname <- file.path(data_dir, "jena_climate_2009_2016.csv")
data <- read_csv(fname)
class(data)

glimpse(data)
dim(data)
head(data)
colnames(data)

library(ggplot2)
ggplot(data, aes(x = 1:nrow(data), y = `T (degC)`)) + geom_line()

data <- data.matrix(data[,-1]) #eliminate first column

#NORMALIZATION
dim(data)
train_data <- data[1:200000,]
mean <- apply(train_data, 2, mean)
std <- apply(train_data, 2, sd)
data <- scale(data, center = mean, scale = std)

head(data)

############################
# THE GENERATOR
############################
#Now here is the data generator:

#    data --> The original array of floating-point data, which you normalized in listing 6.32.
#    lookback --> How many timesteps back the input data should go.
#    delay --> How many timesteps in the future the target should be.
#    min_index and max_index --> Indices in the data array that delimit which timesteps to draw from. This is useful for keeping a segment of the data for validation and another for testing.
#    shuffle --> Whether to shuffle the samples or draw them in chronological order.
#    batch_size --> The number of samples per batch.
#    step --> The period, in timesteps, at which you sample data. You'll set it 6 in order to draw one data point every hour.

generator <- function(data,
                      lookback,
                      delay,
                      min_index,
                      max_index,
                      shuffle = FALSE,
                      batch_size = 128, step = 6) {
  if (is.null(max_index))
    max_index <- nrow(data) - delay - 1
  i <- min_index + lookback
  function() {
    if (shuffle) {
      rows <- sample(c((min_index+lookback):max_index), size = batch_size)
    } else {
      if (i + batch_size >= max_index)
        i <<- min_index + lookback
      rows <- c(i:min(i+batch_size, max_index))
      i <- i + length(rows)
    }
    
    samples <- array(0, dim = c(length(rows), 
                                lookback / step,
                                dim(data)[[-1]]))
    targets <- array(0, dim = c(length(rows)))
                     
    for (j in 1:length(rows)) {
      indices <- seq(rows[[j]] - lookback, rows[[j]], 
                     length.out = dim(samples)[[2]])
      samples[j,,] <- data[indices,]
      targets[[j]] <- data[rows[[j]] + delay,2]
    }            
    
    list(samples, targets)
  }
}
dim(data) # 420551     14
####################
#CUTTING DATA
####################

lookback <- 1440
step <- 6
delay <- 144
batch_size <- 128

train_gen <- generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 1,
  max_index = 200000,
  shuffle = TRUE,
  step = step, 
  batch_size = batch_size
)
dim(samples)
plot(targets)

val_gen = generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 200001,
  max_index = 300000,
  step = step,
  batch_size = batch_size
)
val_gen

test_gen <- generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 300001,
  max_index = NULL,
  step = step,
  batch_size = batch_size
)

# This is how many steps to draw from `val_gen`
# in order to see the whole validation set:
val_steps <- (300000 - 200001 - lookback) / batch_size

  # This is how many steps to draw from `test_gen`
# in order to see the whole test set:
test_steps <- (nrow(data) - 300001 - lookback) / batch_size


#mean(abs(preds - targets))

evaluate_naive_method <- function() {
  batch_maes <- c()
  for (step in 1:val_steps) {
    c(samples, targets) %<-% val_gen()
    preds <- samples[,dim(samples)[[2]],2]
    mae <- mean(abs(preds - targets))
    batch_maes <- c(batch_maes, mae)
  }
  print(mean(batch_maes))
}


library(keras)

model <- keras_model_sequential() %>% 
  layer_gru(units = 32, input_shape = list(NULL, dim(data)[[-1]])) %>% 
  layer_dense(units = 1)

model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "mae"
)

history <- model %>% fit_generator(
  train_gen,
  steps_per_epoch = 500,
  epochs = 20,
  validation_data = val_gen,
  validation_steps = val_steps
)

plot(history)

metrics <- model %>%
  evaluate(test_gen, verbose = 0)

metrics # <- 0.18 wow!
pred2 <- model %>% predict(test_gen) 
