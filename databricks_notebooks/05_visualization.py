import os
import shutil
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.ml.feature import PCA, VectorAssembler, StandardScaler
from pyspark.ml.functions import vector_to_array

# Initialize Spark
builder = SparkSession.builder.appName("IndustrialCluster_Viz")
if os.name == 'nt':
    import sys
    builder = builder.config("spark.driver.bindAddress", "127.0.0.1")
    builder = builder.config("spark.driver.host", "127.0.0.1")
    builder = builder.config("spark.pyspark.python", sys.executable)
    builder = builder.config("spark.pyspark.driver.python", sys.executable)
spark = builder.getOrCreate()

INPUT_PATH = "data/processed/clusters_final"
FIGURES_PATH = "figures"

def run_visualization():
    print("Generating Final Portfolio Visualizations...")
    
    if not os.path.exists(FIGURES_PATH):
        os.makedirs(FIGURES_PATH)
        
    df = spark.read.parquet(INPUT_PATH)
    
    # 1. Prepare for PCA (We need to re-vectorize because we loaded from parquet)
    # The 'features_scaled' column is a VectorUDT, which might need checking.
    # Spark Parquet saves Vectors correctly.
    
    # Use PCA to project to 2D
    pca = PCA(k=2, inputCol="features_scaled", outputCol="pca_features")
    model = pca.fit(df)
    result = model.transform(df)
    
    # Extract X and Y for plotting
    # Convert vector to array
    result = result.withColumn("pca_arr", vector_to_array("pca_features"))
    
    # Collect to Pandas for Plotting (Data is small: 27 UFs)
    pdf = result.select("uf", "cluster_id", "pca_arr", "produtividade_media_uf", "taxa_inovacao_uf").toPandas()
    
    pdf['pca_1'] = pdf['pca_arr'].apply(lambda x: x[0])
    pdf['pca_2'] = pdf['pca_arr'].apply(lambda x: x[1])
    
    # Better Cluster Labels (Simulated mapping for Viz)
    # Ideally correct this based on profiles.csv
    cluster_map = {
        0: "Cluster A (Perfil 1)",
        1: "Cluster B (Perfil 2)",
        2: "Cluster C (Perfil 3)",
        3: "Cluster D (Perfil 4)"
    }
    pdf['Cluster'] = pdf['cluster_id'].map(cluster_map)
    
    # PLOT 1: PCA Scatter
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    scatter = sns.scatterplot(
        data=pdf, 
        x='pca_1', 
        y='pca_2', 
        hue='Cluster', 
        style='Cluster',
        s=150, 
        palette='viridis'
    )
    
    # Annotate points
    for line in range(0, pdf.shape[0]):
         scatter.text(
             pdf.pca_1[line]+0.05, 
             pdf.pca_2[line], 
             pdf.uf[line], 
             horizontalalignment='left', 
             size='medium', 
             color='black', 
             weight='semibold'
         )
         
    plt.title('Mapa de Competitividade Industrial (PCA)', fontsize=16)
    plt.xlabel('Componente Principal 1 (Produtividade/Inovação)', fontsize=12)
    plt.ylabel('Componente Principal 2 (Custo/Intensidade)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    save_path = os.path.join(FIGURES_PATH, "cluster_pca_scatter.png")
    plt.savefig(save_path, dpi=300)
    print(f"PCA Plot saved to {save_path}")
    
    # PLOT 2: Bubble Chart (Productivity vs Innovation)
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=pdf,
        x='produtividade_media_uf',
        y='taxa_inovacao_uf',
        hue='Cluster',
        size='cluster_id', # Just for variety
        sizes=(100, 400),
        palette='deep'
    )
    plt.title('Fronteira Tecnológica: Produtividade vs. Inovação', fontsize=16)
    plt.xlabel('Produtividade do Trabalho (R$/Vínculo)', fontsize=12)
    plt.ylabel('Taxa de Inovação (% do Valor Adicionado)', fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    
    save_path_2 = os.path.join(FIGURES_PATH, "cluster_frontier.png")
    plt.savefig(save_path_2, dpi=300)
    print(f"Frontier Plot saved to {save_path_2}")

if __name__ == "__main__":
    run_visualization()
