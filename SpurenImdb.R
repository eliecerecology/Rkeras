rm(list=ls(all=TRUE))

imdb_dir <- "~/aclImdb"
train_dir <- file.path(imdb_dir, "train")
labels <- c()
texts <- c()

for (label_type in c("neg", "pos")) {
  label <- switch(label_type, neg = 0, pos = 1)
  dir_name <- file.path(train_dir, label_type)
  for (fname in list.files(dir_name, pattern = glob2rx("*.txt"),
                           full.names = TRUE)) {
    texts <- c(texts, readChar(fname, file.info(fname)$size))
    labels <- c(labels, label)
  }
}

library(keras)
maxlen <- 100 #features
training_samples <- 200
validation_samples <- 10000
max_words <- 10000

#break the text 
tokenizer <- text_tokenizer(num_words = max_words) %>%  
  fit_text_tokenizer(texts)
# create index
sequences <- texts_to_sequences(tokenizer, texts)

word_index = tokenizer$word_index # create dictionary
cat("Found", length(word_index), "unique tokens.\n")

data <- pad_sequences(sequences, maxlen = maxlen)
dim(data) #25000 by 100
class(matrix) #matrix
labels <- as.array(labels)
cat("Shape of data tensor:", dim(data), "\n")
cat('Shape of label tensor:', dim(labels), "\n")
indices <- sample(1:nrow(data))

training_indices <- indices[1:training_samples]
validation_indices <- indices[(training_samples + 1):
                              (training_samples + validation_samples)]
x_train <- data[training_indices,]
y_train <- labels[training_indices]
x_val <- data[validation_indices,]
y_val <- labels[validation_indices]

glove_dir = "~/glove/glove.6B"
lines <- readLines(file.path(glove_dir, "glove.6B.100d.txt"))

(lines)[[2]] #it is like a list

embeddings_index <- new.env(hash = TRUE,
                            parent = emptyenv())
for (i in 1:length(lines)) { #400000 lines
  line <- lines[[i]] 
  values <- strsplit(line, " ")[[1]]
  word <- values[[1]] #use the first value 
  embeddings_index[[word]] <- as.double(values[-1])
}
cat("Found", length(embeddings_index), "word vectors.\n")

embedding_dim <- 100
embedding_matrix <- array(0, c(max_words, embedding_dim))
dim(embedding_matrix) # 10000 100

##################
# build an embedding matrix that to load into an embedding layer.
######################
for (word in names(word_index)) { #names(word_index) returns the name of the words
  index <- word_index[[word]]
  if (index < max_words) {
    embedding_vector <- embeddings_index[[word]]
    if (!is.null(embedding_vector))
      embedding_matrix[index+1,] <- embedding_vector           
  }
}

###########################
# model architechture
###########################
model <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words, output_dim = embedding_dim,
                  input_length = maxlen) %>%
  layer_flatten() %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")


summary(model)
get_layer(model, index = 1) %>%
  set_weights(list(embedding_matrix)) %>%
  freeze_weights()

summary(model)

model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("acc")
)
dim(x_train) #200 100
class(x_train) # matrix
dim(y_train) # 200
class(y_train) #array

history <- model %>% fit(
  x_train, y_train,
  epochs = 20,
  batch_size = 32,
  validation_data = list(x_val, y_val)
)
save_model_weights_hdf5(model, "pre_trained_glove_model.h5")
plot(history)

##############################
# model without word embeddig
###############################
model2 <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words, output_dim = embedding_dim,
                  input_length = maxlen) %>%
  layer_flatten() %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")
model2 %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("acc")
)
history2 <- model2 %>% fit(
  x_train, y_train,
  epochs = 20,
  batch_size = 32,
  validation_data = list(x_val, y_val)
)

#Finally, let’s evaluate the model on the test data. First, you need to tokenize the test data.
test_dir <- file.path(imdb_dir, "test")
labels <- c()
texts <- c()

for (label_type in c("neg", "pos")) {
  label <- switch(label_type, neg = 0, pos = 1)
  dir_name <- file.path(test_dir, label_type)
  for (fname in list.files(dir_name, pattern = glob2rx("*.txt"),
                           full.names = TRUE)) {
    texts <- c(texts, readChar(fname, file.info(fname)$size)) #<--------------
    labels <- c(labels, label)
  }
}
sequences <- texts_to_sequences(tokenizer, texts)
x_test <- pad_sequences(sequences, maxlen = maxlen)
y_test <- as.array(labels)

model %>%
  load_model_weights_hdf5("pre_trained_glove_model.h5") %>%
  evaluate(x_test, y_test)



