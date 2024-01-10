from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.sql import Row
import logging
from preprocessing import preprocess
from models import modelTuning
# from src.app.utils import read_file, unzip_file, check_delimiter
from utils import read_file, unzip_file, check_delimiter
import os


def get_spark_session():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(script_dir, "../.."))
    jar_path = os.path.join(project_dir, "config", "postgresql-42.6.0.jar")
    spark = SparkSession.builder \
        .config("spark.jars", jar_path) \
        .config("spark.executor.memory", "8g") \
        .config("spark.executor.cores", "8") \
        .config("spark.cores.max", "8") \
        .master('local[4]') \
        .getOrCreate()
    return spark


if __name__ == '__main__':
    # Set up logging to display only error messages for PySpark
    log = logging.getLogger("pyspark")
    log.setLevel(logging.ERROR)

    folder_path = '../data'
    parquet_output_folder = '../parquet'

    # Create a Spark session using a utility function
    spark = get_spark_session()

    # Get a list of file paths for all CSV files in the folder
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv.bz2')]

    # Raise an exception if no CSV files are found in the specified folder
    if not files:
        raise Exception("No CSV files were found in the provided folder.")

    # Initialize the combined DataFrame with the first DataFrame
    dfs = [preprocess(files[0], spark)]

    # Combine the results of each CSV file into a single DataFrame
    for file in files[1:]:
        # Preprocess each file and select columns matching the first DataFrame
        df = preprocess(file, spark).select(*dfs[0].columns)
        dfs.append(df)

    # Combine DataFrames using union
    combined_data = dfs[0]
    for df in dfs[1:]:
        combined_data = combined_data.union(df)

    # Create the 'parquet' folder if it does not exist
    if not os.path.exists(parquet_output_folder):
        os.makedirs(parquet_output_folder)

    # Save the combined data in Parquet format in the 'parquet' folder
    parquet_output_path = os.path.join(parquet_output_folder, 'combined_data.parquet')

    # Check if the Parquet output path already exists
    if os.path.exists(parquet_output_path):
        print(f"The directory {parquet_output_path} already exists. Performing overwrite.")
        combined_data.write.mode('overwrite').parquet(parquet_output_path)
    else:
        combined_data.write.parquet(parquet_output_path)

    # Call the modelTuning function with the combined data and Spark session
    modelTuning(combined_data, spark)

