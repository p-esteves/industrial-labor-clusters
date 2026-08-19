@echo off
echo Running Industrial Labor Clusters Pipeline...
echo.

REM Check if virtual environment exists and activate it
if exist ".venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
) else (
    echo WARNING: Virtual environment not found at .venv. Using system python.
)

REM Set PySpark Environment Variables explicitly to avoid Windows App Store conflict
REM We use the full path to the python executable in the venv
set "PYSPARK_PYTHON=%CD%\.venv\Scripts\python.exe"
set "PYSPARK_DRIVER_PYTHON=%CD%\.venv\Scripts\python.exe"

echo PySpark Python set to: %PYSPARK_PYTHON%

REM Ensure dependencies are installed
echo Checking dependencies...
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo Failed to install dependencies.
    exit /b %ERRORLEVEL%
)

echo.
echo 1. Generating Data...
"%PYSPARK_DRIVER_PYTHON%" databricks_notebooks/01_ingestion.py
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo.
echo 2. Feature Engineering...
"%PYSPARK_DRIVER_PYTHON%" databricks_notebooks/02_feature_engineering.py
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo.
echo 3. Clustering...
"%PYSPARK_DRIVER_PYTHON%" databricks_notebooks/03_clustering.py
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo.
echo 4. Evaluating...
"%PYSPARK_DRIVER_PYTHON%" databricks_notebooks/04_evaluation.py
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo.
echo 5. Visualization...
"%PYSPARK_DRIVER_PYTHON%" databricks_notebooks/05_visualization.py
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo.
echo Pipeline Completed Successfully!
pause
