# Install and load the necessary libraries
install.packages("dbscan")
library(dbscan)

# Read the data from the CSV file
data <- read.csv("data/samples.csv")

# Get the 2D samples
points <- data[, c("X4.4", "X928")]

# Drop the Null values
points <- na.omit(points)

# Run DBSCAN
dbscan_result <- dbscan(points, eps = 0.5, minPts = 5)

# Print the cluster assignments and number of clusters
cat("Cluster assignments:\n")
print(dbscan_result$cluster)

cat("\nNumber of clusters:", max(dbscan_result$cluster), "\n")

# Visualize the clusters
plot(points, col = dbscan_result$cluster, pch = 19, main = "DBSCAN Clustering")
