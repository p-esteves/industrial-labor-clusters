import os
import shutil
from pyspark.sql import SparkSession

# Initialize Spark
builder = SparkSession.builder.appName("IndustrialCluster_KPIs")
if os.name == 'nt':
    import sys
    builder = builder.config("spark.driver.bindAddress", "127.0.0.1")
    builder = builder.config("spark.driver.host", "127.0.0.1")
    builder = builder.config("spark.pyspark.python", sys.executable)
    builder = builder.config("spark.pyspark.driver.python", sys.executable)
spark = builder.getOrCreate()

FEATURE_DATA_PATH = "data/processed/features_uf_sector"
OUTPUT_PATH = "data/outputs"

def run_kpi_analysis():
    print("Generating Strategic KPI Summary...")
    
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    
    df = spark.read.parquet(FEATURE_DATA_PATH)
    
    # Export to CSV for Tableau/PowerBI or simple Excel verification
    # Rename columns for business users
    df_export = df.selectExpr(
        "uf as UF",
        "round(total_emprego, 0) as Emprego_Industrial",
        "round(total_valor_adicionado / 1000000, 2) as PIB_Industrial_Milhoes",
        "round(produtividade_media_uf, 2) as Produtividade_R_por_Trabalhador",
        "round(custo_trabalho_medio_uf, 2) as Salario_Medio",
        "round(taxa_inovacao_uf * 100, 2) as Inovacao_Pct_VA",
        "round(indice_esg_uf, 1) as Score_ESG"
    ).orderBy("PIB_Industrial_Milhoes", ascending=False)
    
    # Collect to Pandas for single-file CSV save (small data: 27 rows)
    pdf = df_export.toPandas()
    csv_path = os.path.join(OUTPUT_PATH, "kpis_summary.csv")
    pdf.to_csv(csv_path, index=False, sep=";")
    
    print(f"KPI Summary saved to {csv_path}")
    print(pdf.head())

if __name__ == "__main__":
    run_kpi_analysis()
