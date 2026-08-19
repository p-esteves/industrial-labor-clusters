import os
import shutil
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType

# Initialize Spark Session (Robust for Local/Cloud)
builder = SparkSession.builder.appName("IndustrialCluster_Ingestion")
if os.name == 'nt':
    import sys
    builder = builder.config("spark.driver.bindAddress", "127.0.0.1")
    builder = builder.config("spark.driver.host", "127.0.0.1")
    # Force correct python path
    builder = builder.config("spark.pyspark.python", sys.executable)
    builder = builder.config("spark.pyspark.driver.python", sys.executable)
spark = builder.getOrCreate()

RAW_DATA_PATH = "data/raw/industrial_indicators"

def generate_industrial_data():
    """
    Simulates a rich industrial dataset for Brazil's 27 UFs.
    Variables mimic real correlation structures (e.g., High Tech sectors -> High Salary & Innovation).
    """
    print("Generating Industrial Economic Data...")
    
    # 27 UFs
    ufs = [
        'SP', 'RJ', 'MG', 'ES', # Sudeste
        'RS', 'SC', 'PR',       # Sul
        'BA', 'PE', 'CE', 'RN', 'PB', 'SE', 'AL', 'PI', 'MA', # Nordeste
        'DF', 'GO', 'MT', 'MS', # Centro-Oeste
        'AM', 'PA', 'RO', 'RR', 'AP', 'AC', 'TO' # Norte
    ]
    
    # Industrial Sectors (CNAE Groups)
    sectors = [
        "Indústria Extrativa", 
        "Alimentos e Bebidas", 
        "Têxtil e Confecção", 
        "Celulose e Papel",
        "Química e Farmacêutica", 
        "Metalurgia", 
        "Automotiva e Equipamentos", 
        "Máquinas e Aparelhos Elétricos"
    ]
    
    data = []
    
    for uf in ufs:
        # Define structural profile of the UF
        # 1. Industrial Hubs (Mature)
        if uf in ['SP', 'RJ', 'SC', 'RS', 'MG']:
            base_prod = 1.2
            base_salary = 1.3
            tech_bias = 0.3 # Higher chance of tech sectors
        # 2. Agro-Industrial / Emerging
        elif uf in ['MT', 'GO', 'MS', 'PR', 'BA']:
            base_prod = 1.0
            base_salary = 1.0
            tech_bias = 0.0
        # 3. Developing / Extractivist
        else:
            base_prod = 0.7
            base_salary = 0.8
            tech_bias = -0.2

        for sector in sectors:
            # Sector modifiers
            if sector in ["Química e Farmacêutica", "Automotiva e Equipamentos"]:
                sector_mult = 1.5
                energy_intensity = 0.8 # Tech efficiency
            elif sector in ["Indústria Extrativa", "Metalurgia"]:
                sector_mult = 1.2
                energy_intensity = 2.0 # High energy
            else:
                sector_mult = 0.9
                energy_intensity = 1.0
            
            # Generate metrics for 2023
            # Employment (Vínculos)
            base_jobs = 50000 if uf == 'SP' else 5000
            jobs = int(base_jobs * (1 + np.random.normal(0, 0.2)) * sector_mult)
            if jobs < 100: jobs = 100
            
            # Value Added (R$)
            # Productivity = VA / Job
            productivity = 150000 * base_prod * sector_mult * np.random.normal(1, 0.1)
            value_added = jobs * productivity
            
            # Salary Mass (R$)
            # Avg Salary = Base * Sector * UF
            avg_salary = 3000 * base_salary * sector_mult * np.random.normal(1, 0.05)
            salary_mass = jobs * avg_salary
            
            # Energy Consumption (MWh) - Important for "Industry 4.0 vs Traditional"
            energy_consumption = value_added * 0.0005 * energy_intensity * np.random.normal(1, 0.1)
            
            # Innovation Investment (R$)
            # High for Tech sectors in Mature UFs
            innovation_rate = 0.05 if (sector_mult > 1.2 and base_prod > 1.0) else 0.01
            innovation_inv = value_added * innovation_rate * np.random.normal(1, 0.2)
            
            # ESG Index (0-100)
            # Random but correlated with "Modernity"
            esg_score = 60 + (20 * base_prod) + np.random.normal(0, 5)
            esg_score = max(0, min(100, esg_score))
            
            data.append((
                uf,
                sector,
                2023,
                jobs,
                float(value_added),
                float(salary_mass),
                float(energy_consumption),
                float(innovation_inv),
                float(esg_score)
            ))

    schema = StructType([
        StructField("uf", StringType(), True),
        StructField("setor", StringType(), True),
        StructField("ano", IntegerType(), True),
        StructField("emprego_industrial", IntegerType(), True),
        StructField("valor_adicionado_brut", FloatType(), True),
        StructField("massa_salarial", FloatType(), True),
        StructField("consumo_energia_mwh", FloatType(), True),
        StructField("investimento_inovacao", FloatType(), True),
        StructField("indice_esg", FloatType(), True)
    ])

    df = spark.createDataFrame(data, schema)
    
    print(f"Dataset generated: {df.count()} records.")
    df.show(5)
    
    if os.path.exists(RAW_DATA_PATH):
        shutil.rmtree(RAW_DATA_PATH)
    df.write.mode("overwrite").parquet(RAW_DATA_PATH)
    print(f"Saved to {RAW_DATA_PATH}")

if __name__ == "__main__":
    generate_industrial_data()
