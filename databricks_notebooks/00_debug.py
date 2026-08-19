from pyspark.sql import SparkSession
import sys
import os

print(f"Python: {sys.version}")
print(f"Spark Version Info: Checking...")

try:
    spark = SparkSession.builder \
        .appName("Debug_Smoke_Test") \
        .master("local[1]") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .config("spark.driver.host", "127.0.0.1") \
        .config("spark.python.use.daemon", "false") \
        .config("spark.worker.reuse", "false") \
        .getOrCreate()
        
    print(f"Spark Version: {spark.version}")
    
    print("Testing simple DataFrame creation...")
    df = spark.range(10)
    print(f"Count: {df.count()}")
    
    print("Testing collection...")
    rows = df.collect()
    print(f"Collected: {rows}")
    
    print("SUCCESS: Spark environment is working!")

except Exception as e:
    print("FAILURE: Spark environment crashed.")
    print(e)
