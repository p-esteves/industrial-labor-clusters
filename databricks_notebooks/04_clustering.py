import os
import shutil
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml import Pipeline

# Initialize Spark
builder = SparkSession.builder.appName("IndustrialCluster_Model")
if os.name == 'nt':
    import sys
    builder = builder.config("spark.driver.bindAddress", "127.0.0.1")
    builder = builder.config("spark.driver.host", "127.0.0.1")
    builder = builder.config("spark.pyspark.python", sys.executable)
    builder = builder.config("spark.pyspark.driver.python", sys.executable)
spark = builder.getOrCreate()

INPUT_PATH = "data/processed/features_uf_sector"
OUTPUT_PATH = "data/processed/clusters_final"
CSV_OUTPUT = "data/outputs/cluster_profiles.csv"

def run_clustering_pipeline():
    print("Running Unsupervised Learning Pipeline...")
    
    df = spark.read.parquet(INPUT_PATH)
    
    # Select features for Clustering
    # We focus on structural variables (Productivity, Cost, Tech)
    features = [
        "produtividade_media_uf",
        "custo_trabalho_medio_uf",
        "intensidade_energetica_uf",
        "taxa_inovacao_uf", 
        "indice_esg_uf"
    ]
    
    # 1. Pipeline Stages
    assembler = VectorAssembler(inputCols=features, outputCol="features_raw")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features_scaled", withStd=True, withMean=True)
    
    # K-Means (K=4 based on expected profiles: Hub, Emerging, Agro, Lagging)
    kmeans = KMeans(featuresCol="features_scaled", k=4, seed=123, predictionCol="cluster_id")
    
    pipeline = Pipeline(stages=[assembler, scaler, kmeans])
    
    # 2. Fit Model
    model = pipeline.fit(df)
    predictions = model.transform(df)
    
    # 3. Evaluation
    evaluator = ClusteringEvaluator(predictionCol="cluster_id", featuresCol="features_scaled", metricName="silhouette", distanceMeasure="squaredEuclidean")
    silhouette = evaluator.evaluate(predictions)
    print(f"Silhouette Score (Cohesion/Separation): {silhouette:.4f}")
    
    # 4. Profiling the Clusters (Interpretation)
    print("Generating Cluster Profiles...")
    
    # Save predictions
    if os.path.exists(OUTPUT_PATH):
        shutil.rmtree(OUTPUT_PATH)
    predictions.write.mode("overwrite").parquet(OUTPUT_PATH)
    
    # Create Interpretation CSV (Group By Cluster)
    df_profile = predictions.groupBy("cluster_id").avg(
        "produtividade_media_uf", 
        "custo_trabalho_medio_uf", 
        "taxa_inovacao_uf",
        "indice_esg_uf"
    ).orderBy("cluster_id")
    
    pdf_profile = df_profile.toPandas()
    
    # Labeling Logic (Automated Interpretation)
    def label_cluster(row):
        # Heuristic rules based on the means
        if row['avg(produtividade_media_uf)'] > 200000: # Very high productivity
            return "1. Polo Industrial Avançado (Maduro)"
        elif row['avg(taxa_inovacao_uf)'] < 0.015:
            return "4. Baixo Dinamismo / Tradicional"
        elif row['avg(custo_trabalho_medio_uf)'] > 3500:
             return "2. Alto Custo / Estagnado"
        else:
            return "3. Emergente / Agro-Indústria"

    # Note: Thresholds above are illustrative, in a real run we'd inspect the centers.
    # For now we save the raw means.
    
    pdf_profile.to_csv(CSV_OUTPUT, index=False)
    print(f"Cluster Profiles saved to {CSV_OUTPUT}")

if __name__ == "__main__":
    run_clustering_pipeline()
