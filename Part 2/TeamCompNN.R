library(keras)
library(tfruns)
library(rsample)

# Splitting Data into Training Sets
split <- initial_split(data.NN$data.temp, prop = 4/5)
train <- training(split)
test <- testing(split)

# Setting up Vectorization Layer
num_words <- 160 # 157 Champions, some extra for white space and extra characters
max_length <- 5  # 5 Champions per team
text_vectorization <- layer_text_vectorization(
  max_tokens = num_words,
  output_sequence_length = max_length
)

text_vectorization %>% 
  adapt(data.NN$data.temp$team)

input <- layer_input(shape = c(1), dtype = "string")

FLAGS <- flags(
  flag_numeric("dropout", 0.3),
  flag_numeric("unit", 8)
)

output <- input %>% 
  text_vectorization() %>% 
  layer_embedding(input_dim = num_words + 1, output_dim = FLAGS$unit) %>% 
  layer_global_average_pooling_1d() %>% 
  layer_dense(units = 3*FLAGS$unit, activation = "tanh") %>% 
  layer_dropout(rate = FLAGS$dropout) %>% 
  layer_dense(units = 2*FLAGS$unit) %>% 
  layer_dropout(rate = FLAGS$dropout) %>% 
  layer_dense(units = FLAGS$unit, activation = "relu") %>% 
  layer_dropout(rate = FLAGS$dropout) %>% 
  layer_dense(units = 1, activation = "sigmoid")

model <- keras_model(input, output)

model %>% 
  compile(
    optimizer = "adam",
    loss = "binary_crossentropy",
    metrics = list("accuracy")
  )

history <- model %>% 
  fit(
    train$team,
    as.numeric(train$win == TRUE),
    epochs = 15,
    batch_size = 512,
    validation_split = 0.2,
    verbose = 2
  )

results <- model %>% 
  evaluate(test$team, as.numeric(test$win == TRUE), verbose = 0)

plot(history)