from pyspark.sql import SparkSession
from pyspark.sql import Row

from src.app.utils import read_file, unzip_file, check_delimiter
import os


def get_spark_session():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(script_dir, "../.."))
    jar_path = os.path.join(project_dir, "config", "postgresql-42.6.0.jar")

    spark = SparkSession.builder.config("spark.jars", jar_path).master('local[4]').getOrCreate()

    return spark


if __name__ == '__main__':
    # src/data/2008.csv.bz2
    # spark = SparkSession.builder.config("spark.jars", "postgresql-42.6.0.jar").master('local[4]').getOrCreate()
    spark = get_spark_session()
    # unzip_file('../data/2008.csv.bz2')
    # check_delimiter('../data/2008.csv')
    ruta_hdfs = "hdfs://0.0.0.0:9000/data/datanode/"

    read_file(ruta_hdfs, spark)
