library(keras)
library(tfruns)
library(rsample)

# Splitting Data into Training Sets
split <- initial_split(data.NN, prop = 4/5)
train <- training(split)
test <- testing(split)

# Vectorization Layer
num_words <- 160
max_length <- 5

text_vectorization <- layer_text_vectorization(
  max_tokens = num_words,
  output_sequence_length = max_length
)

text_vectorization %>% 
  adapt(data.raw$team) # Using data.raw for all teams

input_1 <- layer_input(shape = c(1), dtype = "string")
input_2 <- layer_input(shape = c(1), dtype = "string")

FLAGS <- flags(
  flag_numeric("dropout", 0.3),
  flag_numeric("unit", 16)
)

text_embedding <- layer_embedding(input_dim = num_words + 1, output_dim = FLAGS$unit)

embedding_1 <- input_1 %>% 
  text_vectorization() %>% 
  text_embedding()

embedding_2 <- input_2 %>% 
  text_vectorization %>% 
  text_embedding()

output <- layer_concatenate(c(embedding_1, embedding_2), axis=2) %>% 
  layer_global_average_pooling_1d() %>% 
  layer_dense(units = 3*FLAGS$unit, activation = "tanh") %>% 
  layer_dropout(rate = FLAGS$dropout) %>% 
  layer_dense(units = 2*FLAGS$unit, activation = "gelu") %>% 
  layer_dropout(rate = FLAGS$dropout) %>% 
  layer_dense(units = FLAGS$unit, activation = "relu") %>% 
  layer_dropout(rate = FLAGS$dropout) %>% 
  layer_dense(units = 1, activation = "sigmoid")

model <- keras_model(inputs = c(input_1, input_2), output)

model %>% 
  compile(
    optimizer = "adam",
    loss = "binary_crossentropy",
    metrics = list("accuracy")
  )

history <- model %>% 
  fit(
    list(train$team_1, train$team_2),
    as.numeric(train$win == TRUE),
    epochs = 20,
    batch_size = 10000,
    validation_split = 0.2,
    verbose = 2
  )

results1 <- model %>% 
  evaluate(list(test$team_1, test$team_2), as.numeric(test$win == TRUE), verbose = 0)

plot(history)

