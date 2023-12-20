from pyspark.sql import SparkSession
import bz2
import pandas as pd


def unzip_file(input_path):
    output_path = '../data/2008.csv'

    with bz2.BZ2File(input_path, 'rb') as source, open(output_path, 'wb') as dest:
        dest.write(source.read())


def check_delimiter(file_path, num_lines=5):
    with open(file_path, 'r', encoding='ISO-8859-1') as file:
        for _ in range(num_lines):
            line = file.readline()
            print(line.strip())


def read_file(file_path, spark):
    # pandas_df = pd.read_csv(file_path)
    # print(pandas_df.head())

    csv_options = {
        "header": True,
        "inferSchema": True,
        "encoding": "UTF-8",
        "delimiter": ",",
    }

    raw_df = spark.read.format("csv").options(**csv_options).load(file_path)

    return raw_df
