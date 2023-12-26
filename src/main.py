from time import time
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vectors
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import udf
from pyspark.sql import functions as F


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

# Step 4 : Execute the k-means algorithm
# Step 4.1 : Merge the two features into one VectorUDT feature
# Based on :
# https://stackoverflow.com/questions/46710934/pyspark-sql-utils-illegalargumentexception-ufield-features-does-not-exist
features_col = ['x_vectorized_scaled', 'y_vectorized_scaled']
assembler = VectorAssembler(inputCols=features_col, outputCol="merged_vectorized")
points = assembler.transform(points)

# Step 4.2 : Execute the k-Means algorithm
kmeans = KMeans(featuresCol='merged_vectorized', predictionCol="Cluster").setK(5).setSeed(1)
model = kmeans.fit(points)
points_clustered = model.transform(points)


# Step 5 : Outlier Detection
distFromCenter = udf()
points_clustered = points_clustered.withColumn("distance_from_center",
                                               distance_udf("merged_vectorized", model.clusterCenters()))
# points_clustered.show(5)

# Stop Spark Context
spark.stop()

# Stop the timer that captures the execution time of main function
main_end = time()

# Print the execution time of Spark program
print("Execution Time of Main = ", main_end - main_start)
