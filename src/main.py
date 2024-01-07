# Import the necessary libraries
import sys
from time import time
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vectors
from pyspark.sql.types import DoubleType, ArrayType
from pyspark.sql.functions import udf, col

# Setup for asking the dataset path in spark-submit
if len(sys.argv) != 2:
    print("Usage: spark-submit main.py <dataset_path>")
    sys.exit(1)

dataset_path = sys.argv[1]

# Start the timer that captures the execution time of main function
main_start = time()

# Initialize a Spark Session
spark = SparkSession.builder \
    .appName("DataMiningProject") \
    .getOrCreate()

# Step 1 : Read the dataset and store it into a DataFrame
# Specify that the CSV does not have a header and let Spark infer the Schema of it
points = spark.read.csv(dataset_path, header=False, inferSchema=True)

# Step 1.1 : Rename the columns to x and y respectfully
points = points \
    .withColumnRenamed("_c0", "x") \
    .withColumnRenamed("_c1", "y")

# Step 2 : Clear records that do not have and the two fields
points = points.na.drop()

# Step 3 : Scale each column to 0-1
# Step 3.1 : Transform Double Type points to VectorUDT for the MinMaxScaler
for column in points.columns:
    # Convert the column to a VectorUDT type
    assembler = VectorAssembler(inputCols=[column], outputCol=f"{column}_vectorized")
    points = assembler.transform(points)
columns = ['x_vectorized', 'y_vectorized']
# Step 3.2 : Scale each column to 0-1
for column in columns:
    # Create a MinMaxScaler for each column
    scaler = MinMaxScaler(inputCol=column, outputCol=f"{column}_scaled")

    # Fit and transform the DataFrame
    scaler_model = scaler.fit(points)
    points = scaler_model.transform(points)

# Step 3.4 : Drop the vectorized non-scaled points
points = points.drop(*['x_vectorized', 'y_vectorized'])

# Step 4 : Execute the k-means algorithm
# Step 4.1 : Merge the two features into one VectorUDT feature
# Based on :
# https://stackoverflow.com/questions/46710934/pyspark-sql-utils-illegalargumentexception-ufield-features-does-not-exist
features_col = ['x_vectorized_scaled', 'y_vectorized_scaled']
assembler = VectorAssembler(inputCols=features_col, outputCol="merged_vectorized")
points = assembler.transform(points)

# Step 4.2 : Execute the k-Means algorithm
kmeans = KMeans(featuresCol='merged_vectorized', predictionCol="Cluster").setK(35).setSeed(4)
model = kmeans.fit(points)
points_clustered = model.transform(points)

# Step 4.3 : Drop the scaled vectorized columns
points_clustered = points_clustered.drop(*['x_vectorized_scaled', 'y_vectorized_scaled'])

# Step 5 : Outlier Detection using distance from center and z-score

# Step 5.1 : Add the coordinates of each center to initial DataFrame
# Extract cluster centers
centers = model.clusterCenters()
# Create a UDF that maps the index of the cluster to its center
centersCoord_udf = udf(lambda cluster: list(map(float, centers[cluster])), ArrayType(DoubleType()))
points_clustered = points_clustered.withColumn("Center", centersCoord_udf(col("Cluster")))

# Step 5.2 : Calculate distance for each point to its assigned cluster center
distance_udf = udf(lambda features, center: float(Vectors.squared_distance(features, center)), DoubleType())
points_clustered = points_clustered.withColumn("distance_to_cluster_center",
                                               distance_udf(col("merged_vectorized"), col("center")))

# Step 5.3: Calculate the mean and standard deviation of column distance_to_cluster_center
mean = points_clustered.agg({"distance_to_cluster_center": "mean"}).collect()[0][0]
stddev = points_clustered.agg({"distance_to_cluster_center": "stddev"}).collect()[0][0]

# Step 5.4: Compute the z-score of column distance_to_cluster_center using a User Defined Function
z_score_udf = udf(lambda center: abs((center - mean) / stddev), DoubleType())

# Step 5.5: Add a new column with the z-score
points_clustered = points_clustered.withColumn("z_score", z_score_udf(col("distance_to_cluster_center")))

# Step 5.6: Set a z-score threshold
z_threshold = 5.4

# Step 5.7: Identify outliers using the z-score threshold
outliers = points_clustered.filter(col("z_score") > z_threshold)

# Step 5.7a : If a dataset does not have such a high z-score samples.
# Using quantiles find a z-score threshold that is 99.9% furthest.
if outliers.isEmpty():
    q_threshold = 0.999
    z_threshold = points_clustered.approxQuantile("z_score", [q_threshold], 0.0)[0]
    outliers = points_clustered.filter(col("z_score") > z_threshold)

# Step 5.8: Drop unnecessary columns and show the outliers
outliers.drop(*['merged_vectorized', 'Cluster', 'Center', 'distance_to_cluster_center', 'z_score']).show(outliers.count(), truncate=False)

# Stop the timer that captures the execution time of main function
main_end = time()

# Print the execution time of Spark program
print("Execution Time of Main = ", main_end - main_start)

# Stop Spark Context
spark.stop()
