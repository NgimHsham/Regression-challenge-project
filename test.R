install.packages("readr")  # Install 'readr' package if not already installed
library(readr)

# Read the CSV files
predicted <- read_csv("0075710_NGIM.csv")
labels <- read_csv("test_completo_ch.csv")

# Extract the relevant columns
predicted_knn <- predicted$pred_knn
predicted_lm <- predicted$pred_lm
actual_labels <- labels$Y

# Calculate the Mean Squared Error (MSE)
mse_knn <- mean((actual_labels - predicted_knn)^2)
mse_lm<- mean((actual_labels - predicted_lm)^2)

# Print the MSE
print(mse_knn)
print(mse_knn)
