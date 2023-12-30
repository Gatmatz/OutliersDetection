# Read the data from the CSV file
data <- read.csv("data/samples.csv")

# Get the 2D samples
points <- data[, c("X4.4", "X928")]

# Drop the Null values
points <- na.omit(points)

# Specify the number of clusters (in this case, 5)
num_clusters <- 5

# Run K-means clustering
kmeans_result <- kmeans(points, centers = num_clusters)

# Print the cluster assignments and number of clusters
cat("Cluster assignments:\n")
print(kmeans_result$cluster)

cat("\nNumber of clusters:", num_clusters, "\n")

# Visualize the clusters
plot(points, col = kmeans_result$cluster, pch = 19, main = "K-means Clustering")