# Load necessary libraries
library(tidyverse)
library(ggplot2)

# Read the data from the CSV file
data <- read.csv("data/initial_data.csv")

# Min-max scale the samples to [0, 1]
min_max_scale <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}

# Scale to every column separately
scaled_data <- data
scaled_data[, 1:2] <- apply(data[, 1:2], 2, min_max_scale)

# Read the initial centers from Spark Clustering
initial_centers = read.csv("data/coordinates.csv")

# Run k-means on scaled centers
kmeans_result <- kmeans(initial_centers[,2:3], centers = 5)

# Assign clusters to the scaled centers
initial_centers$new_cluster <- kmeans_result$cluster

# Extract distinct combinations of old and new clusters
matching <- distinct(initial_centers, Cluster, new_cluster)

# Merge the dataset with the matching dataframe
merged_dataset <- merge(scaled_data, matching, by = "Cluster")

# Select only the relevant columns
result_dataset <- merged_dataset[, c("x", "y", "new_cluster")]

# Rename the columns for clarity
colnames(result_dataset) <- c("X", "Y", "new_cluster")

# Get the centers of the kmeans clustering
centers <- as.data.frame(kmeans_result$centers)

ggplot(scaled_data, aes(x = x, y = y)) +
  geom_point(size = 3, color="blue") +
  labs(title = "Final k-Means", x = "X", y = "Y") +
  
  # Overlay k-means centers in yellow
  geom_point(data = as.data.frame(centers), aes(x = X, y = Y), color = "yellow", size = 3) +
  
  theme_minimal()