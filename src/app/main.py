from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.sql import Row
import logging
from preprocessing import preprocess
from models import modelTuning
#from src.app.utils import read_file, unzip_file, check_delimiter
from utils import read_file, unzip_file, check_delimiter



if __name__ == '__main__':
    log = logging.getLogger("pyspark")
    log.setLevel(logging.ERROR)
    spark = SparkSession.builder \
        .config("spark.jars", "postgresql-42.6.0.jar") \
        .config("spark.executor.memory", "8g") \
        .config("spark.executor.cores", "8") \
        .config("spark.cores.max", "8") \
        .master('local[4]') \
        .getOrCreate()
    cleanData = preprocess('../data/2008.csv.bz2', spark)
    modelTuning(cleanData, spark)

