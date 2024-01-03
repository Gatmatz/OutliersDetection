from time import time
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vectors
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import udf, col
import pandas as pd
import matplotlib.pyplot as plt

# Ask for the dataset path
filename = input('Give the dataset filename:')

# Start the timer that captures the execution time of main function
main_start = time()

# Initialize a Spark Session
spark = SparkSession.builder \
    .appName("DataMiningProject") \
    .getOrCreate()

# Step 1 : Read the dataset and store it into a DataFrame
# Specify that the CSV does not have a header and let Spark infer the Schema of it
dataset_path = '../data/' + filename
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

# Step 3.2 : Drop the initial points
points = points.drop(*['x', 'y'])

# Step 3.3 : Scale each column to 0-1
for column in points.columns:
    # Create a MinMaxScaler for each column
    scaler = MinMaxScaler(inputCol=column, outputCol=f"{column}_scaled")

    # Fit and transform the DataFrame
    scaler_model = scaler.fit(points)
    points = scaler_model.transform(points)

# Step 3.4 : Drop the vectorized non-scaled points
points = points.drop(*['x_vectorized', 'y_vectorized'])

# Save the normalization to a Pandas DataFrame for scattering
pandas_df = points.toPandas()
pandas_df['x'] = pandas_df['x_vectorized_scaled'].apply(lambda x: x[0])
pandas_df['y'] = pandas_df['y_vectorized_scaled'].apply(lambda x: x[0])

# Step 4 : Execute the k-means algorithm
# Step 4.1 : Merge the two features into one VectorUDT feature
# Based on :
# https://stackoverflow.com/questions/46710934/pyspark-sql-utils-illegalargumentexception-ufield-features-does-not-exist
features_col = ['x_vectorized_scaled', 'y_vectorized_scaled']
assembler = VectorAssembler(inputCols=features_col, outputCol="merged_vectorized")
points = assembler.transform(points)

# Step 4.2 : Execute the k-Means algorithm
kmeans = KMeans(featuresCol='merged_vectorized', predictionCol="Cluster").setK(15).setSeed(3)
model = kmeans.fit(points)
points_clustered = model.transform(points)
points_clustered = points_clustered.drop(*['x_vectorized_scaled', 'y_vectorized_scaled'])

# Step 5 : Outlier Detection using distance from center and Percentiles
# Step 5.2 : Add the coordinates of each cluster to initial DataFrame
centers = model.clusterCenters()  # Extract cluster centers
centers_df = spark.createDataFrame([(i, Vectors.dense(center)) for i, center in enumerate(centers)],
                                   ["center_id", "center"])  # Create a DataFrame with cluster centers

points_distance = points_clustered \
    .join(centers_df, on=points_clustered['Cluster'] == centers_df[
    'center_id'])  # Join the original DataFrame with the cluster centers DataFrame

# Step 5.3 : Calculate distance for each point to its assigned cluster center
distance_udf = udf(lambda features, center: float(Vectors.squared_distance(features, center)), DoubleType())
points_distance = points_distance.withColumn("distance_to_cluster_center",
                                             distance_udf(col("merged_vectorized"), col("center")))

# Step 5.4: Set a threshold based on percentiles
percentile_threshold = 99

# Calculate the threshold value based on the specified percentile
threshold_value = points_distance.approxQuantile("distance_to_cluster_center", [percentile_threshold / 100], 0.0)[0]

# Identify outliers using the percentile threshold
outliers = points_distance.filter(col("distance_to_cluster_center") > threshold_value)
# outliers = outliers.orderBy(F.desc("distance_to_cluster_center")).limit(10)

# Show the outliers
outliers.show(truncate=False)


# Scatter plot the initial points, the cluster centers and the outliers
centers = model.clusterCenters()
pandas_centers_df = pd.DataFrame(centers, columns=["x", "y"])
outliers_df = pd.DataFrame(outliers.toPandas(), columns=['merged_vectorized'])
outliers_df[['x', 'y']] = outliers_df['merged_vectorized'].apply(pd.Series)
outliers_df = outliers_df.drop('merged_vectorized', axis=1)
plt.scatter(pandas_df["x"], pandas_df["y"])
plt.scatter(pandas_centers_df["x"], pandas_centers_df["y"], marker="X", color="red")
plt.scatter(outliers_df["x"], outliers_df["y"], marker="X", color="yellow")
plt.title("KMeans Cluster Centers and Outlier Detection")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.show()

# Stop Spark Context
spark.stop()

# Stop the timer that captures the execution time of main function
main_end = time()

# Print the execution time of Spark program
print("Execution Time of Main = ", main_end - main_start)