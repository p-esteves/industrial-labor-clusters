@echo off
setlocal EnableDelayedExpansion

REM 1. Activate Virtual Environment
call .venv\Scripts\activate.bat

REM 2. PATHS CONFIGURATION (SUBST TRICK)
REM Spark hates spaces in paths ("Pietro Esteves"). We map to T: drive.
subst T: "%~dp0."
if exist T:\run_portable.bat (
    T:
    cd \
) else (
    echo WARNING: SUBST failed, running in original path...
)

set "PROJECT_ROOT=%CD%\"
set "HADOOP_HOME=%PROJECT_ROOT%hadoop"

REM --- PORTABLE JAVA CONFIGURATION (The Magic Fix) ---
set "JAVA_HOME=%PROJECT_ROOT%java_portable"
REM Check if portable java exists
if not exist "%JAVA_HOME%\bin\java.exe" (
    echo ERROR: Portable Java not found. Please run setup_portable_java.ps1
    pause
    exit /b 1
)
set "PATH=%JAVA_HOME%\bin;%HADOOP_HOME%\bin;%PATH%"

REM 3. Python Path Handling (Short Path for spaces)
set "FULL_PYTHON=%PROJECT_ROOT%.venv\Scripts\python.exe"
for %%I in ("%FULL_PYTHON%") do set "SHORT_PYTHON=%%~sI"
set "PYSPARK_PYTHON=%SHORT_PYTHON%"
set "PYSPARK_DRIVER_PYTHON=%SHORT_PYTHON%"

REM 4. Spark Configs (tuned for local file execution)
REM Note: We don't need 'spark.driver.host' hacks if Java is correct!
REM But we keep local[1] for safety on personal laptop.

echo ==================================================
echo  INDUSTRIAL LABOR CLUSTERS - PORTABLE EXECUTION
echo ==================================================
echo  JAVA: Using Portable OpenJDK 11 (Safe Mode)
echo  PYTHON: %SHORT_PYTHON%
echo  HADOOP: %HADOOP_HOME%
echo ==================================================

echo.
echo [1/5] Running Ingestion (Economic Simulation)...
"%SHORT_PYTHON%" databricks_notebooks/01_ingestion_simulation.py > portable_1_ingestion.log 2>&1
if %ERRORLEVEL% NEQ 0 ( 
    echo FAILED at Ingestion. See portable_1_ingestion.log
    type portable_1_ingestion.log
    exit /b 1 
)

echo.
echo [2/5] Running Feature Engineering (Productivity/Innovation)...
"%SHORT_PYTHON%" databricks_notebooks/02_feature_engineering.py > portable_2_feature.log 2>&1
if %ERRORLEVEL% NEQ 0 ( 
    echo FAILED at Feature Engineering. See portable_2_feature.log
    type portable_2_feature.log
    exit /b 1 
)

echo.
echo [3/5] Running KPI Analysis (CSV Export)...
"%SHORT_PYTHON%" databricks_notebooks/03_kpi_analysis.py > portable_3_kpi.log 2>&1
if %ERRORLEVEL% NEQ 0 ( 
    echo FAILED at KPI Analysis. See portable_3_kpi.log
    type portable_3_kpi.log
    exit /b 1 
)

echo.
echo [4/5] Running Clustering (K-Means Pipeline)...
"%SHORT_PYTHON%" databricks_notebooks/04_clustering.py > portable_4_clustering.log 2>&1
if %ERRORLEVEL% NEQ 0 ( 
    echo FAILED at Clustering. See portable_4_clustering.log
    type portable_4_clustering.log
    exit /b 1 
)

echo.
echo [5/5] Running Visualization (PCA & Frontier)...
"%SHORT_PYTHON%" databricks_notebooks/05_visualization.py > portable_5_viz.log 2>&1
if %ERRORLEVEL% NEQ 0 ( 
    echo FAILED at Visualization. See portable_5_viz.log
    type portable_5_viz.log
    exit /b 1 
)

echo.
echo ==================================================
echo  SUCCESS! CHECK 'figures/' FOLDER.
echo ==================================================
