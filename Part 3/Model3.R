library(keras)
library(tfruns)
library(rsample)

# For this model, I'm going to add an additional input which includes tier of gameplay

# Splitting Data into Training Sets
split <- initial_split(data.NN.tier, prop = 4/5)
train <- training(split)
test <- testing(split)

# Vectorization Layer
num_words <- 160
max_length <- 5

text_vectorization <- layer_text_vectorization(
  max_tokens = num_words,
  output_sequence_length = max_length
)
text_vectorization_tier <- layer_text_vectorization(
  max_tokens = 8,
  output_sequence_length = 1
)

text_vectorization %>% 
  adapt(data.raw$team) # Using data.raw for all teams
text_vectorization_tier %>% 
  adapt(data.NN.tier$tier)


input_1 <- layer_input(shape = c(1), dtype = "string")
input_2 <- layer_input(shape = c(1), dtype = "string")
input_3 <- layer_input(shape = c(1), dtype = "string")

FLAGS <- flags(
  flag_numeric("dropout", 0.5),
  flag_numeric("unit", 8)
)

text_embedding <- layer_embedding(input_dim = num_words + 1, output_dim = 5)

embedding_1 <- input_1 %>% 
  text_vectorization() %>% 
  text_embedding() 

embedding_2 <- input_2 %>% 
  text_vectorization() %>% 
  text_embedding() 

embedding_3 <- input_3 %>% 
  text_vectorization_tier() %>% 
  text_embedding() 


output <- layer_concatenate(c(embedding_1, embedding_2, embedding_3), axis=-1) %>% # Need to figure out how to concatenate this properly
  layer_global_average_pooling_1d() %>% 
  layer_dense(units = FLAGS$unit, activation = "relu") %>% 
  layer_dropout(rate = FLAGS$dropout) %>% 
  layer_dense(units = 1, activation = "sigmoid")

model <- keras_model(inputs = c(input_1, input_2, input_3), output)

model %>% 
  compile(
    optimizer = "Adam",
    loss = "binary_crossentropy",
    metrics = list("accuracy")
  )

history <- model %>% 
  fit(
    list(train$team_1, train$team_2, train$tier),
    as.numeric(train$win == TRUE),
    epochs = 25,
    batch_size = 1000,
    validation_split = 0.2,
    verbose = 2
  )

results <- model %>% 
  evaluate(list(test$team_1, test$team_2, test$tier), as.numeric(test$win == TRUE), verbose = 0)

plot(history)

