from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml import Pipeline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Initialize Spark Session
# Initialize Spark Session (Robust Windows/Databricks dual-mode)
builder = SparkSession.builder.appName("ObservatorioIndustrial_Evaluation")
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

def evaluate_clusters():
    print("Starting Model Evaluation (Elbow Method & Silhouette)...")
    
    df = spark.read.parquet(INDEX_DATA_PATH)
    
    features = [
        "avg_growth_rate", 
        "volatility_index", 
        "avg_salary_nominal", 
        "productivity_proxy", 
        "diversification_index"
    ]
    
    assembler = VectorAssembler(inputCols=features, outputCol="features_raw")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=True)
    
    pipeline_prep = Pipeline(stages=[assembler, scaler])
    model_prep = pipeline_prep.fit(df)
    df_prepared = model_prep.transform(df)
    
    evaluator = ClusteringEvaluator(predictionCol="prediction", featuresCol="features", metricName="silhouette", distanceMeasure="squaredEuclidean")
    
    results = []
    
    print(f"{'K':<5} | {'Silhouette':<10} | {'WSSSE (Cost)':<20}")
    print("-" * 40)
    
    for k in range(2, 7):
        kmeans = KMeans(featuresCol="features", k=k, seed=42)
        model = kmeans.fit(df_prepared)
        predictions = model.transform(df_prepared)
        
        silhouette = evaluator.evaluate(predictions)
        cost = model.summary.trainingCost
        
        results.append({"k": k, "silhouette": silhouette, "cost": cost})
        print(f"{k:<5} | {silhouette:<10.4f} | {cost:<20.4f}")

    # Plotting Logic using Pandas/Matplotlib
    pdf = pd.DataFrame(results)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia (WSSSE)', color=color)
    ax1.plot(pdf['k'], pdf['cost'], color=color, marker='o', label='WSSSE')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Silhouette Score', color=color)
    ax2.plot(pdf['k'], pdf['silhouette'], color=color, marker='s', linestyle='--', label='Silhouette')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Cluster Evaluation: Elbow Method & Silhouette')
    plt.tight_layout()
    plt.savefig("figures/evaluation_metrics.png")
    print("\nEvaluation plot saved to figures/evaluation_metrics.png")

if __name__ == "__main__":
    evaluate_clusters()
