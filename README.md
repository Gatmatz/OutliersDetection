# Spark Outlier Detection with K-means Algorithm

## Overview

This project utilizes Apache Spark for outlier detection on 2D point data, employing the K-means clustering algorithm. Outlier detection is crucial in identifying rare instances or anomalies within a dataset. The K-means algorithm is leveraged to group normal data points, and outliers are identified based on their deviation from the established clusters.

## Features

- **Spark Integration:** Harness the power of Apache Spark for distributed computing, enabling scalability for large datasets.
- **K-means Algorithm:** Utilize the K-means clustering algorithm to group similar data points and identify outliers.
- **2D Point Analysis:** Focus on outlier detection within 2D point data, suitable for various applications such as fraud detection, network security, or sensor data analysis.
- **Scalable:** Designed to scale seamlessly with Spark, making it suitable for processing massive datasets in parallel.

## Getting Started

### 1. Clone the Repository:

```bash
git clone https://github.com/Gatmatz/spark-outlier-detection.git
```

### 2. Install Conda Environment:

```bash
conda env create -f environment.yml
```

### 3. Run the Spark Job:

```bash
spark-submit <main.py path> <dataset path>
```
