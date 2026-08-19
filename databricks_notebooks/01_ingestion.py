import os
import shutil
import random
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType
from pyspark.sql import functions as F

# Initialize Spark Session
# Initialize Spark Session (Robust Windows/Databricks dual-mode)
builder = SparkSession.builder.appName("ObservatorioIndustrial_Ingestion")
if os.name == 'nt':
    import sys
    builder = builder.config("spark.driver.bindAddress", "127.0.0.1")
    builder = builder.config("spark.driver.host", "127.0.0.1")
    # Force correct python path (ignoring env var issues)
    builder = builder.config("spark.pyspark.python", sys.executable)
    builder = builder.config("spark.pyspark.driver.python", sys.executable)
spark = builder.getOrCreate()

# Configuration
RAW_DATA_PATH = "data/raw/rais_simulated"
YEARS = [2019, 2020, 2021, 2022, 2023]
UFS = [
    # Sudeste
    'SP', 'RJ', 'MG', 'ES',
    # Sul
    'RS', 'SC', 'PR',
    # Centro-Oeste
    'DF', 'GO', 'MT', 'MS',
    # Nordeste
    'BA', 'PE', 'CE', 'RN', 'PB', 'SE', 'AL', 'PI', 'MA',
    # Norte
    'AM', 'PA', 'RO', 'RR', 'AP', 'AC', 'TO'
]

SECTORS = ["Metalmecânica", "Química e Petroquímica", "Alimentos e Bebidas", "Têxtil e Vestuário", "Construção Civil"]

def generate_synthetic_data():
    """
    Generates synthetic data mimicking RAIS attributes with embedded economic profiles
    to ensure cluster separability:
    1. Mature: High Salary, Stable Growth (Low Volatility)
    2. Emerging: Medium Salary, High Growth
    3. Stagnant: Low Salary, Low/Negative Growth
    """
    data = []
    
    print("Generating synthetic data...")
    
    for uf in UFS:
        # Define Economic Profile based on UF (simplified for demonstration)
        if uf in ['SP', 'RJ', 'SC', 'DF', 'PR', 'RS']:
            profile = 'MATURE'
            base_salary = np.random.normal(3500, 500)
            growth_trend = 1.02  # 2% annual growth (steady)
        elif uf in ['MT', 'GO', 'MS', 'PE', 'BA', 'CE']:
            profile = 'EMERGING'
            base_salary = np.random.normal(2400, 400)
            growth_trend = 1.08  # 8% annual growth (rapid)
        else:
            profile = 'STAGNANT'
            base_salary = np.random.normal(1800, 300)
            growth_trend = 1.005 # 0.5% growth (stagnant)

        for setor in SECTORS:
            # Sector modifiers
            if setor == "Química e Petroquímica":
                sector_salary_mult = 1.5
            elif setor == "Têxtil e Vestuário":
                sector_salary_mult = 0.8
            else:
                sector_salary_mult = 1.0

            current_jobs = int(np.random.normal(10000, 2000)) if profile == 'MATURE' else int(np.random.normal(4000, 1000))

            for ano in YEARS:
                # Apply growth trend with some noise
                current_jobs = int(current_jobs * growth_trend * np.random.normal(1.0, 0.02))
                
                # Monthly seasonality simulation
                for mes in range(1, 13):
                    # Economic shock in 2020 (Pandemic)
                    shock = 0.90 if ano == 2020 and mes > 3 else 1.0
                    recovery = 1.05 if ano == 2021 else 1.0
                    
                    # Random monthly fluctuation
                    monthly_jobs = int(current_jobs * shock * recovery * np.random.normal(1.0, 0.01))
                    
                    # Calculate stats
                    avg_salary = (base_salary * sector_salary_mult * (1.05 ** (ano - 2019))) # 5% nominal inflation
                    total_salary_mass = monthly_jobs * avg_salary
                    
                    data.append((
                        uf, 
                        ano, 
                        mes, 
                        setor, 
                        monthly_jobs, 
                        float(avg_salary),
                        float(total_salary_mass)
                    ))

    # Create DataFrame
    schema = StructType([
        StructField("uf", StringType(), True),
        StructField("ano", IntegerType(), True),
        StructField("mes", IntegerType(), True),
        StructField("setor", StringType(), True),
        StructField("empregos", IntegerType(), True),
        StructField("salario_medio_nominal", FloatType(), True),
        StructField("massa_salarial", FloatType(), True)
    ])

    df = spark.createDataFrame(data, schema)
    
    print(f"Dataset generated with {df.count()} records.")
    
    # Save to Parquet
    if os.path.exists(RAW_DATA_PATH):
        shutil.rmtree(RAW_DATA_PATH)
    
    df.write.mode("overwrite").parquet(RAW_DATA_PATH)
    print(f"Data saved to {RAW_DATA_PATH}")

if __name__ == "__main__":
    generate_synthetic_data()
