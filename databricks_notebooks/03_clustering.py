from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA
from pyspark.ml.clustering import KMeans
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, udf
from pyspark.sql.types import FloatType
import os
import shutil

# Initialize Spark Session
# Initialize Spark Session (Robust Windows/Databricks dual-mode)
builder = SparkSession.builder.appName("ObservatorioIndustrial_Clustering")
if os.name == 'nt':
    import sys
    builder = builder.config("spark.driver.bindAddress", "127.0.0.1")
    builder = builder.config("spark.driver.host", "127.0.0.1")
    builder = builder.config("spark.python.use.daemon", "false")
    builder = builder.config("spark.python.worker.reuse", "false")
    builder = builder.config("spark.pyspark.python", sys.executable)
    builder = builder.config("spark.pyspark.driver.python", sys.executable)
spark = builder.getOrCreate()

INDEX_DATA_PATH = "data/processed/features_uf"
CLUSTERS_DATA_PATH = "data/processed/clusters"

def run_clustering():
    print("Starting Clustering...")
    
    df = spark.read.parquet(INDEX_DATA_PATH)
    
    # Select features for clustering
    # We use meaningful economic features
    features = [
        "avg_growth_rate", 
        "volatility_index", 
        "avg_salary_nominal", 
        "productivity_proxy", 
        "diversification_index"
    ]
    
    print(f"Features selected: {features}")
    
    # ML Pipeline
    assembler = VectorAssembler(inputCols=features, outputCol="features_raw")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=True)
    
    # PCA for Visualization (k=2)
    pca = PCA(k=2, inputCol="features", outputCol="pca_features")
    
    # K-Means
    # We choose k=3 based on our theoretical segments: Mature, Emerging, Stagnant
    kmeans = KMeans(featuresCol="features", k=3, seed=42)
    
    pipeline = Pipeline(stages=[assembler, scaler, pca, kmeans])
    
    model = pipeline.fit(df)
    predictions = model.transform(df)
    
    print("Clustering completed. Cluster Centers:")
    centers = model.stages[-1].clusterCenters()
    for center in centers:
        print(center)
        
    # Extract PCA components for plotting
    # PCA results are in 'pca_features' column (Vector). We need to split into x and y.
    
    first_element = udf(lambda v: float(v[0]), FloatType())
    second_element = udf(lambda v: float(v[1]), FloatType())
    
    predictions = predictions.withColumn("pca_x", first_element("pca_features")) \
                             .withColumn("pca_y", second_element("pca_features"))

    # Save predictions (drop heavy vector columns)
    final_df = predictions.drop("features_raw", "features", "pca_features")
    
    print("Preview with clusters:")
    final_df.select("uf", "prediction", "avg_growth_rate", "avg_salary_nominal").show(5)
    
    if os.path.exists(CLUSTERS_DATA_PATH):
        shutil.rmtree(CLUSTERS_DATA_PATH)
        
    final_df.write.mode("overwrite").parquet(CLUSTERS_DATA_PATH)
    print(f"Clusters saved to {CLUSTERS_DATA_PATH}")

if __name__ == "__main__":
    run_clustering()
