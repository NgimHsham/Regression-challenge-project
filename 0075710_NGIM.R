# Load required libraries
library(ggplot2)
library(tibble)
library(tidyr)


# Read the train CSV file
train_data <- read.csv("./Regression/train_ch.csv")
train_data$ID <- NULL
test_data_challenge <- read.csv("./Regression/test_ch.csv")
test_data_challenge <- test_data_challenge[, -1]


# Summary statistics
summary(train_data)

####################################################################
# Scatter Plot
####################################################################
# Load required libraries
library(ggplot2)
library(gridExtra)

# Define feature names
feature_names <- paste0("v", 1:9)

# Create a list to store the scatter plots
scatter_plots <- list()

# Generate scatter plots
for (i in 1:length(feature_names)) {
  scatter_plots[[i]] <- ggplot(train_data, aes(x = .data[[feature_names[i]]], y = Y)) +
    geom_point() +
    labs(x = feature_names[i], y = "Y", title = paste("Scatter Plot:", feature_names[i])) +
    theme(plot.title = element_text(hjust = 0.5))
}

# Combine the scatter plots into one figure
combined_plot <- grid.arrange(grobs = scatter_plots, nrow = 3, ncol = 3)

# Save the combined figure
ggsave("combined_scatter_plots.png", combined_plot, width = 10, height = 10, dpi = 300)

####################################################################
# Correlation matrix
####################################################################

library(corrplot)
cor_matrix <- cor(train_data[, c(feature_names, label)])
corrplot(cor_matrix, method = "circle")
####################################################################
####################################################################
# Load required libraries
library(ggplot2)
library(patchwork)

# Define feature names and label
feature_names <- paste0("v", 1:9)
label <- "Y"

# Create a list to store scatter plots
scatter_plots <- list()

# Scatter plots with interaction effect
for (i in 1:(length(feature_names) - 1)) {
  for (j in (i + 1):length(feature_names)) {
    scatter_plot <- ggplot(train_data, aes(x = .data[[feature_names[i]]], y = .data[[label]], color = .data[[feature_names[j]]])) +
      geom_point() +
      labs(x = feature_names[i], y = label, title = paste("Scatter Plot:", feature_names[i], "vs", label, "Stratified by", feature_names[j])) +
      theme(plot.title = element_text(hjust = 0.5))
    
    scatter_plots[[length(scatter_plots) + 1]] <- scatter_plot
  }
}

# Combine scatter plots into one figure
combined_plots <- scatter_plots[[1]]
for (k in 2:length(scatter_plots)) {
  combined_plots <- combined_plots + scatter_plots[[k]]
}

# Save combined scatter plot image
ggsave("combined_scatter_plots.png", combined_plots, width = 30, height = 24, dpi = 300)


####################################################################
# Machine Learning
####################################################################

# Function to evaluate model using multiple metrics
evaluate_model <- function(model, validation_data, test_data) {
  # RMSE
  val_predictions <- predict(model, newdata = validation_data)
  val_rmse <- sqrt(mean((val_predictions - validation_data$Y)^2))
  cat("RMSE - Validation:", val_rmse, "\n")
  
  test_predictions <- predict(model, newdata = test_data)
  test_rmse <- sqrt(mean((test_predictions - test_data$Y)^2))
  cat("RMSE - Test:", test_rmse, "\n")
  
  # Mean Absolute Error (MAE)
  val_mae <- mean(abs(val_predictions - validation_data$Y))
  cat("MAE - Validation:", val_mae, "\n")
  
  test_mae <- mean(abs(test_predictions - test_data$Y))
  cat("MAE - Test:", test_mae, "\n")
  
  # R-squared
  val_r_squared <- 1 - sum((validation_data$Y - val_predictions)^2) / sum((validation_data$Y - mean(validation_data$Y))^2)
  cat("R-squared - Validation:", val_r_squared, "\n")
  
  test_r_squared <- 1 - sum((test_data$Y - test_predictions)^2) / sum((test_data$Y - mean(test_data$Y))^2)
  cat("R-squared - Test:", test_r_squared, "\n")
  
  # Adjusted R-squared
  n <- nrow(validation_data)
  p <- length(coef(model))
  val_adj_r_squared <- 1 - (1 - val_r_squared) * ((n - 1) / (n - p - 1))
  cat("Adjusted R-squared - Validation:", val_adj_r_squared, "\n")
  
  test_adj_r_squared <- 1 - (1 - test_r_squared) * ((nrow(test_data) - 1) / (nrow(test_data) - p - 1))
  cat("Adjusted R-squared - Test:", test_adj_r_squared, "\n")
}



# Load required libraries
library(caret)

# Split the data 60-20-20
set.seed(123)  # for reproducibility
train_index <- createDataPartition(train_data$Y, p = 0.6, list = FALSE)
train_set <- train_data[train_index, ]
remaining_data <- train_data[-train_index, ]

validation_index <- createDataPartition(remaining_data$Y, p = 0.5, list = FALSE)
validation_set <- remaining_data[validation_index, ]
test_set <- remaining_data[-validation_index, ]

######################################################################
# Train and test the LR model on the original features only 
######################################################################
# Train Linear Regression model

# Print the updated dataframe
head(test_set)

lm_model <- lm(Y ~ ., data = train_set)

evaluate_model(lm_model, validation_set, test_set)

##################33
# Perform feature engineering (example: create interaction terms)
# Function to perform feature engineering and create interaction terms
perform_feature_engineering <- function(train_set, validation_set, test_set, test_data_challenge) {
  # Define the feature names
  feature_names <- paste0("v", 1:9)
  
  # Loop through each feature and create interaction terms
  for (feature in feature_names) {
    # Create interaction term with v2
    train_set[[paste0(feature, "IntV2")]] <- train_set[[feature]] * train_set$v2
    validation_set[[paste0(feature, "IntV2")]] <- validation_set[[feature]] * validation_set$v2
    test_set[[paste0(feature, "IntV2")]] <- test_set[[feature]] * test_set$v2
    test_data_challenge[[paste0(feature, "IntV2")]] <- test_data_challenge[[feature]] * test_data_challenge$v2
    
    # Create interaction term with v3
    train_set[[paste0(feature, "IntV3")]] <- train_set[[feature]] * train_set$v3
    validation_set[[paste0(feature, "IntV3")]] <- validation_set[[feature]] * validation_set$v3
    test_set[[paste0(feature, "IntV3")]] <- test_set[[feature]] * test_set$v3
    test_data_challenge[[paste0(feature, "IntV3")]] <- test_data_challenge[[feature]] * test_data_challenge$v3
    
    # Create interaction term with v4
    train_set[[paste0(feature, "IntV4")]] <- train_set[[feature]] * train_set$v4
    validation_set[[paste0(feature, "IntV4")]] <- validation_set[[feature]] * validation_set$v4
    test_set[[paste0(feature, "IntV4")]] <- test_set[[feature]] * test_set$v4
    test_data_challenge[[paste0(feature, "IntV4")]] <- test_data_challenge[[feature]] * test_data_challenge$v4
    
    # Create interaction term with v5
    train_set[[paste0(feature, "IntV5")]] <- train_set[[feature]] * train_set$v5
    validation_set[[paste0(feature, "IntV5")]] <- validation_set[[feature]] * validation_set$v5
    test_set[[paste0(feature, "IntV5")]] <- test_set[[feature]] * test_set$v5
    test_data_challenge[[paste0(feature, "IntV5")]] <- test_data_challenge[[feature]] * test_data_challenge$v5
    
    # Create interaction term with v6
    train_set[[paste0(feature, "IntV6")]] <- train_set[[feature]] * train_set$v6
    validation_set[[paste0(feature, "IntV6")]] <- validation_set[[feature]] * validation_set$v6
    test_set[[paste0(feature, "IntV6")]] <- test_set[[feature]] * test_set$v6
    test_data_challenge[[paste0(feature, "IntV6")]] <- test_data_challenge[[feature]] * test_data_challenge$v6
    
    # Create interaction term with v7
    train_set[[paste0(feature, "IntV7")]] <- train_set[[feature]] * train_set$v7
    validation_set[[paste0(feature, "IntV7")]] <- validation_set[[feature]] * validation_set$v7
    test_set[[paste0(feature, "IntV7")]] <- test_set[[feature]] * test_set$v7
    test_data_challenge[[paste0(feature, "IntV7")]] <- test_data_challenge[[feature]] * test_data_challenge$v7
    
    # Create interaction term with v8
    train_set[[paste0(feature, "IntV8")]] <- train_set[[feature]] * train_set$v8
    validation_set[[paste0(feature, "IntV8")]] <- validation_set[[feature]] * validation_set$v8
    test_set[[paste0(feature, "IntV8")]] <- test_set[[feature]] * test_set$v8
    test_data_challenge[[paste0(feature, "IntV8")]] <- test_data_challenge[[feature]] * test_data_challenge$v8
    
    # Create interaction term with v9
    train_set[[paste0(feature, "IntV9")]] <- train_set[[feature]] * train_set$v9
    validation_set[[paste0(feature, "IntV9")]] <- validation_set[[feature]] * validation_set$v9
    test_set[[paste0(feature, "IntV9")]] <- test_set[[feature]] * test_set$v9
    test_data_challenge[[paste0(feature, "IntV9")]] <- test_data_challenge[[feature]] * test_data_challenge$v9
  }
  
  # Return the modified datasets
  return(list(train_set = train_set, validation_set = validation_set, test_set = test_set, test_data_challenge = test_data_challenge))
}

########
# Function to perform non/linear transformations on features
perform_nonlinear_transformations <- function(data) {
  # Define the feature names
  feature_names <- paste0("v", 1:9)
  
  # Loop through each feature and apply non/linear transformations
  for (feature in feature_names) {
    
    # Apply power of 2 transformation
    data[[paste0(feature, "_pow2")]] <- data[[feature]]^2
    
    # Apply power of 3 transformation
    data[[paste0(feature, "_pow3")]] <- data[[feature]]^3
    # Apply power of 4 transformation
    data[[paste0(feature, "_pow4")]] <- data[[feature]]^4
    # Apply power of 5 transformation
    data[[paste0(feature, "_pow5")]] <- data[[feature]]^5
    
  }
  
  # Return the modified dataset
  return(data)
}



########### 1
# Call the function to perform feature engineering
result <- perform_feature_engineering(train_set, validation_set, test_set, test_data_challenge)

# Access the modified datasets
train_set_Interaction <- result$train_set
validation_set_Interaction <- result$validation_set
test_set_Interaction <- result$test_set
test_data_challenge_Interaction <- result$test_data_challenge
########## 2
# Call the function to perform nonlinear transformations 
train_set_nonlinear <- perform_nonlinear_transformations(train_set_Interaction)
validation_set_nonlinear <- perform_nonlinear_transformations(validation_set_Interaction)
test_set_nonlinear <- perform_nonlinear_transformations(test_set_Interaction)
test_data_challenge_nonlinear <- perform_nonlinear_transformations(test_data_challenge_Interaction)




# Randomly shuffle the training data
train_set_nonlinear <- train_set_nonlinear[sample(nrow(train_set_nonlinear)), ]


# Train Linear Regression model
lm_model <- lm(Y ~ ., data = train_set_nonlinear)
# Evaluate models
evaluate_model(lm_model, validation_set_nonlinear, test_set_nonlinear)


#########
###################
###########
# KNN
library(FNN)
# Function to evaluate model using multiple metrics
evaluate_knn_model <- function(train_data, validation_data, test_data, best_k) {
  
  # RMSE
  val_predictions <- knn.reg(validation_data[, -c(1, ncol(validation_data))], train=train_data[, -c(1, ncol(train_data))], y=train_data$Y, k = best_k)$pred
  val_rmse <- sqrt(mean((val_predictions - validation_data$Y)^2))
  cat("RMSE - Validation:", val_rmse, "\n")
  
  test_predictions <- knn.reg(test_data[, -c(1, ncol(test_data))], train=train_data[, -c(1, ncol(train_data))], y=train_data$Y, k = best_k)$pred
  test_rmse <- sqrt(mean((test_predictions - test_data$Y)^2))
  cat("RMSE - Test:", test_rmse, "\n")
  
  # Mean Absolute Error (MAE)
  val_mae <- mean(abs(val_predictions - validation_data$Y))
  cat("MAE - Validation:", val_mae, "\n")
  
  test_mae <- mean(abs(test_predictions - test_data$Y))
  cat("MAE - Test:", test_mae, "\n")
  
  # R-squared
  val_r_squared <- 1 - sum((validation_data$Y - val_predictions)^2) / sum((validation_data$Y - mean(validation_data$Y))^2)
  cat("R-squared - Validation:", val_r_squared, "\n")
  
  test_r_squared <- 1 - sum((test_data$Y - test_predictions)^2) / sum((test_data$Y - mean(test_data$Y))^2)
  cat("R-squared - Test:", test_r_squared, "\n")
  
}

train_model <- function(best_k, train_data, val_data, test_data) {
  # Train the model
  model <- knn.reg(train=train_data[, -c(1, ncol(train_data))], y=train_data$Y, k=best_k)
  
  # Predict using the model
  predictions_val <- knn.reg(val_data[, -c(1, ncol(val_data))], train=train_data[, -c(1, ncol(train_data))], y=train_data$Y, k = best_k)$pred
  predictions_test <- knn.reg(test_data[, -c(1, ncol(test_data))], train=train_data[, -c(1, ncol(train_data))], y=train_data$Y, k = best_k)$pred
  
  # Calculate RMSE
  rmse_val <- sqrt(mean((val_data$Y - predictions_val)^2))
  rmse_test <- sqrt(mean((test_data$Y - predictions_test)^2))
  
  # Print evaluation metric
  print(paste("RMSE val:", rmse_val))
  print(paste("RMSE test:", rmse_test))
  
  # Return model
  return(model)
}

find_best_k <- function(max_k, train_data, val_data, test_data) {
  
  best_k <- NULL
  best_rmse <- Inf
  
  # Iterate over possible k values
  for (k in 1:max_k) { # Adjust the range based on your problem
    # Train the model
    model <- knn.reg(train=train_data[, -c(1, ncol(train_data))], y=train_data$Y, k=k)
    
    # Predict using the model
    predictions_val <- knn.reg(val_data[, -c(1, ncol(val_data))], train=train_data[, -c(1, ncol(train_data))], y=train_data$Y, k = k)$pred
    
    # Calculate RMSE
    rmse_val <- sqrt(mean((val_data$Y - predictions_val)^2))
    
    # If the current model has the lowest RMSE seen so far, update the best_k and best_rmse
    if (rmse_val < best_rmse) {
      best_k <- k
      best_rmse <- rmse_val
    }
  }
  
  # Train the final model using the best_k
  model <- knn.reg(train=train_data[, -c(1, ncol(train_data))], y=train_data$Y, k=best_k)
  
  # Predict using the final model
  predictions_test <- knn.reg(test_data[, -c(1, ncol(test_data))], train=train_data[, -c(1, ncol(train_data))], y=train_data$Y, k = best_k)$pred
  
  # Calculate RMSE for the test set
  rmse_test <- sqrt(mean((test_data$Y - predictions_test)^2))
  
  # Print evaluation metric
  print(paste("Best k:", best_k))
  print(paste("RMSE val:", best_rmse))
  print(paste("RMSE test:", rmse_test))
  
  # Return model
  return(best_k)
}


# Call the train_model function
max_k <- 20
##### Train with original features
best_k <- find_best_k(max_k, train_set, validation_set, test_set)
knn_model <- train_model(best_k, train_set, validation_set, test_set)
evaluate_knn_model(train_set, validation_set, test_set, best_k)
##### Train with original as well as engineered features
best_k <- find_best_k(max_k, train_set_nonlinear, validation_set_nonlinear, test_set_nonlinear)
knn_model <- train_model(best_k, train_set_nonlinear, validation_set_nonlinear, test_set_nonlinear)
evaluate_knn_model(train_set_nonlinear, validation_set_nonlinear, test_set_nonlinear, best_k)

#############################
# FEATURE SELECTION
#############################
# Load necessary libraries
library(caret)
library(corrplot)

# Define a function for feature selection based on correlation
feature_selection <- function(train_data, threshold) {
  correlations <- cor(train_data)
  correlated_features <- findCorrelation(correlations, cutoff = threshold)
  selected_features <- colnames(train_data)[-correlated_features]
  return(selected_features)
}

# Select features with correlation above 0.5 (for instance)
selected_features <- feature_selection(train_set_nonlinear, 0.5)
# Remove 'ID' from the selected features
selected_features <- selected_features[selected_features != "ID"]
# Print the updated selected features
print(selected_features)


# Use only the selected features in the train, validation and test sets
train_set_selected <- train_set_nonlinear[, c(selected_features, "Y")]
validation_set_selected <- validation_set_nonlinear[, c(selected_features, "Y")]
test_set_selected <- test_set_nonlinear[, c(selected_features, "Y")]

# Call the train_model function with the selected features
knn_model <- train_model(best_k, train_set_selected, validation_set_selected, test_set_selected)


################
# Test the challenge test data
#####
# Read the test CSV file

# Function to get predictions from a model
get_lm_preds <- function(model, test_data) {
  test_predictions <- predict(model, newdata = test_data)
  return(test_predictions)
}

get_knn_preds <- function(train_data, test_data, best_k) {
  test_predictions <- knn.reg(test_data, train = train_data[, -which(names(train_data) == "Y")], y = train_data$Y, k = best_k)$pred
  return(test_predictions)
}


# Get linear regression predictions for the test data
lm_preds <- get_lm_preds(lm_model, test_data_challenge_nonlinear)
knn_preds <- get_knn_preds(train_set_nonlinear, test_data_challenge_nonlinear, best_k)

# Create a data frame with the predictions
predictions <- data.frame(pred_knn = knn_preds, pred_lm = lm_preds)

# Save the predictions to a CSV file
write.csv(predictions, file = "./Regression/0075710_NGIM.csv", row.names = FALSE)

