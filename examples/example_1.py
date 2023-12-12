from pyspark.sql import SparkSession
from pyspark.sql import Row

spark = SparkSession.builder.config("spark.jars", "postgresql-42.6.0.jar").master('local[4]').getOrCreate()

# Some data with their column names. With DF we can structure our data
columns = ["id", "name", "surname", "age", "country", "local_phone"]
input_data = [(1, "Simón", "Bolivar", 47, "VEN", "489 895 965"),
              (2, "Fidel", "Castro", 90, "CU", "956 268 348"),
              (3, "Jose", "Doroteo", 45, "MEX", "985 621 444"),
              (4, "Ernesto", "Guevara", 39, "AR", "895 325 481"),
              (5, "Hugo", "Chávez", 58, "VE", "489 895 965"),
              (6, "Camilo", "Cienfuegos", 27, "CUB", "956 268 348"),
              (7, "Emiliano", "Zapata", 39, "ME", "985 621 444"),
              (8, "Juan Domingo", "Perón", 78, "ARG", "985 621 444"),
              ]

# Simplier data
int_list = [1, 2, 3]

# intDF = spark.createDataFrame(int_list).toDF("value") # this doesnt work
# DF from primitive indicating type
intDF = spark.createDataFrame(int_list, "int").toDF("value")
intDF.printSchema()
intDF.show()

complexDF = spark.createDataFrame(input_data)
complexDF.printSchema()
complexDF.show()
