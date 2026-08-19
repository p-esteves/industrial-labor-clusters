import os
import shutil
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Initialize Spark
builder = SparkSession.builder.appName("IndustrialCluster_Features")
if os.name == 'nt':
    import sys
    builder = builder.config("spark.driver.bindAddress", "127.0.0.1")
    builder = builder.config("spark.driver.host", "127.0.0.1")
    builder = builder.config("spark.pyspark.python", sys.executable)
    builder = builder.config("spark.pyspark.driver.python", sys.executable)
spark = builder.getOrCreate()

RAW_DATA_PATH = "data/raw/industrial_indicators"
FEATURE_DATA_PATH = "data/processed/features_uf_sector"

def run_feature_engineering():
    print("Calculating Economic Indicators...")
    
    df = spark.read.parquet(RAW_DATA_PATH)
    
    # 1. Fundamental KPIs
    # Productivity: Value Added per Employee
    # Labor Cost: Salary Mass per Employee (Proxy for Avg Salary but structurally)
    # Energy Intensity: Energy / Value Added (Efficiency)
    # Innovation Rate: Investment / Value Added
    
    df_features = df.withColumn("produtividade_trabalho", F.col("valor_adicionado_brut") / F.col("emprego_industrial")) \
                    .withColumn("custo_medio_trabalho", F.col("massa_salarial") / F.col("emprego_industrial")) \
                    .withColumn("intensidade_energetica", F.col("consumo_energia_mwh") / F.col("valor_adicionado_brut")) \
                    .withColumn("taxa_inovacao", F.col("investimento_inovacao") / F.col("valor_adicionado_brut"))
    
    # 2. Sectoral Aggregation per UF (Weighted Averages)
    # We want to cluster UFs, so we need to aggregate the sector data into a UF profile.
    # We can use weighted averages by Employment or Value Added.
    
    df_uf = df_features.groupBy("uf").agg(
        F.sum("emprego_industrial").alias("total_emprego"),
        F.sum("valor_adicionado_brut").alias("total_valor_adicionado"),
        
        # Weighted Average of Productivity (Weighted by Jobs)
        (F.sum(F.col("produtividade_trabalho") * F.col("emprego_industrial")) / F.sum("emprego_industrial")).alias("produtividade_media_uf"),
        
        # Weighted Average of Salary (Weighted by Jobs)
        (F.sum(F.col("custo_medio_trabalho") * F.col("emprego_industrial")) / F.sum("emprego_industrial")).alias("custo_trabalho_medio_uf"),
        
        # Weighted Average of Energy Intensity (Weighted by VA)
        (F.sum(F.col("intensidade_energetica") * F.col("valor_adicionado_brut")) / F.sum("valor_adicionado_brut")).alias("intensidade_energetica_uf"),
        
        # Weighted Innovation (Weighted by VA)
        (F.sum(F.col("taxa_inovacao") * F.col("valor_adicionado_brut")) / F.sum("valor_adicionado_brut")).alias("taxa_inovacao_uf"),
        
        # ESG is a simple average
        F.avg("indice_esg").alias("indice_esg_uf")
    )
    
    print("Analysis Preview (UF Level):")
    df_uf.select("uf", "produtividade_media_uf", "taxa_inovacao_uf").show(5)
    
    if os.path.exists(FEATURE_DATA_PATH):
        shutil.rmtree(FEATURE_DATA_PATH)
    df_uf.write.mode("overwrite").parquet(FEATURE_DATA_PATH)
    print(f"Features saved to {FEATURE_DATA_PATH}")

if __name__ == "__main__":
    run_feature_engineering()
