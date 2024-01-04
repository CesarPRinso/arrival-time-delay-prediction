from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.sql import Row
import logging
from preprocessing import preprocess
from src.app.utils import read_file, unzip_file, check_delimiter
import logging
import os

if __name__ == '__main__':
    log = logging.getLogger("pyspark")
    log.setLevel(logging.ERROR)
    spark = SparkSession.builder \
        .config("spark.jars", "postgresql-42.6.0.jar") \
        .config("spark.executor.memory", "4g") \
        .config("spark.executor.cores", "4") \
        .config("spark.cores.max", "4") \
        .master('local[4]') \
        .getOrCreate()
    preprocess('../data/2008.csv.bz2', spark)
