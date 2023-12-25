from pyspark.sql import SparkSession
from time import time
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
import numpy as np
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType

# Ask for the dataset path
filename = input('Give the dataset filename:')

# Start the timer that captures the execution time of main function
main_start = time()

# Initialize a Spark Session
spark = SparkSession.builder\
    .appName("DataMiningProject")\
    .getOrCreate()

# Step 1 : Read the dataset and store it into a DataFrame
# Specify that the CSV does not have a header and let Spark infer the Schema of it
dataset_path = '../data/' + filename
points = spark.read.csv(dataset_path, header=False, inferSchema=True)

# Step 1.5 : Rename the columns to x and y respectfully
points = points.\
    withColumnRenamed("_c0", "x").\
    withColumnRenamed("_c1", "y")

# Step 2 : Clear records that do not have and the two fields
points = points.na.drop()

# Step 3 : Scale each column to 0-1
# Step 3.1 : Transform Double Type points to VectorUDT for the MinMaxScaler
for column in points.columns:
    # Convert the column to a VectorUDT type
    assembler = VectorAssembler()\
        .setInputCols([column])\
        .setOutputCol(f"{column}_vectorized")

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


# Show the dataset
points.show(5)

# Stop Spark Context
spark.stop()

# Stop the timer that captures the execution time of main function
main_end = time()

# Print the execution time of Spark program
print("Execution Time of Main = ", main_end-main_start)
