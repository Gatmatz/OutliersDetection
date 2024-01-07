# Install and load the necessary libraries
install.packages("dbscan")
library(dbscan)

# Read the data from the CSV file
data <- read.csv("data/samples.csv")

# Drop the Null values
data <- na.omit(data)

# Min-max scale the samples to [0, 1]
min_max_scale <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}

# Scale to every column separately
scaled_data <- apply(data, 2, min_max_scale)

# Run DBSCAN
dbscan_result <- dbscan(scaled_data, eps = 0.01, minPts = 5)

cat("\nNumber of clusters:", max(dbscan_result$cluster), "\n")

# Visualize the clusters
plot(scaled_data, col = dbscan_result$cluster, pch = 19, main = "DBSCAN Clustering in R")