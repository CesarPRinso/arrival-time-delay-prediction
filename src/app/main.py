import sys

from app.model.models import modelTuning, compile_models

sys.path.append("..")
from pyspark.sql import SparkSession
import logging
from preprocessing import preprocess
import os
import time


def get_spark_session():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(script_dir, "../.."))
    jar_path = os.path.join(project_dir, "config", "postgresql-42.6.0.jar")
    spark = SparkSession.builder \
        .config("spark.jars", jar_path) \
        .config("spark.executor.memory", "16g") \
        .config("spark.executor.cores", "16") \
        .config("spark.cores.max", "16") \
        .config("spark.ui.reverseProxy", "true") \
        .master('local[8]') \
        .getOrCreate()
    return spark


def load_files(folder_path, infer_schema=True):
    # Get a list of file paths for all CSV files in the folder
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv.bz2')]

    # Raise an exception if no CSV files are found in the specified folder
    if not files:
        raise Exception("No CSV files were found in the provided folder.")

    # Read the first file to get the schema
    first_df = preprocess(files[0], spark)

    # Initialize the combined DataFrame with the first DataFrame
    dfs = [first_df]

    # Combine the results of each CSV file into a single DataFrame
    for file in files[1:]:
        # Preprocess each file and select columns matching the first DataFrame
        df = preprocess(file, spark).select(*dfs[0].columns)
        dfs.append(df)

    # Combine DataFrames using union
    combined_data = dfs[0]
    for df in dfs[1:]:
        combined_data = combined_data.union(df)

    return combined_data


def write_to_parquet(dataframe, parquet_output_folder):
    # Create the 'parquet' folder if it does not exist
    if not os.path.exists(parquet_output_folder):
        os.makedirs(parquet_output_folder)

    # Save the combined data in Parquet format in the 'parquet' folder
    parquet_output_path = os.path.join(parquet_output_folder, 'combined_data.parquet')

    # Check if the Parquet output path already exists
    if os.path.exists(parquet_output_path):
        print(f"The directory {parquet_output_path} already exists. Performing overwrite.")
        dataframe.write.mode('overwrite').parquet(parquet_output_path)
    else:
        dataframe.write.parquet(parquet_output_path)

    return dataframe


if __name__ == '__main__':
    # Set up logging to display only error messages for PySpark
    print('..... start')
    log = logging.getLogger("pyspark")
    log.setLevel(logging.ERROR)
    if len(sys.argv) > 1:
        selected_model = int(sys.argv[1])
        if selected_model in [1, 2, 3, 4]:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            folder_path = os.path.join(current_dir, '../data')
            parquet_output_folder = os.path.join(current_dir, '../parquet')

            # Create a Spark session using a utility function
            spark = get_spark_session()
            start_time = time.time()
            combined_data = load_files(folder_path)
            write_to_parquet(combined_data, parquet_output_folder)
            elapsed_time = time.time() - start_time
            print(f"Execution time: {elapsed_time} seconds")
            # Call the modelTuning function with the combined data and Spark session
            compile_models(combined_data, selected_model, spark)
            spark.stop()
        else:
            # Invalid option, prompt the user to enter a valid option
            print("Invalid option. Please provide a valid option (1, 2, 3, or 4) for model training.")
    else:
        # No option provided, prompt the user to enter a valid option
        print("No option provided. Please provide a valid option (1, 2, 3, or 4) for model training.")
