from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.sql import Row
import logging
from preprocessing import preprocess
from src.app.utils import read_file, unzip_file, check_delimiter



if __name__ == '__main__':
    # src/data/2008.csv.bz2
    # Configura el nivel de registro de log para Spark
    log = logging.getLogger("pyspark")
    log.setLevel(logging.ERROR)
    spark = SparkSession.builder.config("spark.jars", "postgresql-42.6.0.jar").master('local[4]').getOrCreate()
    # unzip_file('../data/2008.csv.bz2')
    # check_delimiter('../data/2008.csv')
    #read_file('../data/2008.csv.bz2', spark)
    preprocess('../data/2008.csv.bz2', spark)



