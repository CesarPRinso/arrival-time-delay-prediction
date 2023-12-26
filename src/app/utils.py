from pyspark.sql import SparkSession
import bz2
from pyspark.sql import functions as F
from pyspark.sql.connect.functions import when
from pyspark.sql.types import StringType
from pyspark.sql.functions import col


def analysis_file(cleaned_df):
    output_file = "column_analysis.txt"
    # Open the file in write mode
    with open(output_file, "w") as file:
        for column_name in cleaned_df.columns:
            data_type = cleaned_df.schema[column_name].dataType

            # Get the distinct values for the column
            distinct_values_count = cleaned_df.select(column_name).distinct().count()

            if distinct_values_count <= 1:
                # Column has a single value, write to file and drop the column
                file.write(f"Column '{column_name}' has a single value for all rows. It will be dropped.\n")
                cleaned_df = cleaned_df.drop(column_name)

            elif distinct_values_count <= 10:
                file.write(f"Analysis of the column '{column_name}':\n")

                if isinstance(data_type, StringType):
                    # Get the frequency of each value
                    frequencies = cleaned_df.groupBy(column_name).count()

                    file.write(frequencies.toPandas().to_string(index=False) + "\n")

                    unique_values_count = cleaned_df.select(column_name).distinct().count()
                    file.write(f"Total unique values: {unique_values_count}\n")

                    unique_values = cleaned_df.select(column_name).distinct().collect()
                    file.write("Unique values:\n")
                    for value in unique_values:
                        file.write(f"{value[column_name]}\n")

                    file.write("\n")

                else:
                    total_values_count = cleaned_df.select(column_name).count()
                    unique_values_count = cleaned_df.select(column_name).distinct().count()

                    file.write(f"  - Data type: {data_type}\n")
                    file.write(f"  - Total values: {total_values_count}\n")
                    file.write(f"  - Total unique values: {unique_values_count}\n")

                    unique_values = cleaned_df.select(column_name).distinct().collect()
                    file.write("Unique values:\n")
                    for value in unique_values:
                        file.write(f"{value[column_name]}\n")

                    file.write("\n")
            else:
                file.write(f"Analysis of the column '{column_name}' (top 10 values):\n")
                top_values = cleaned_df.groupBy(column_name).count().orderBy(col("count").desc()).limit(10)
                file.write(top_values.toPandas().to_string(index=False) + "\n\n")

    # Print the file path
    print(f"Results saved to: {output_file}")
    return cleaned_df


def delete(raw_df):
    total_rows_before = raw_df.count()
    print(f"Total rows before: {total_rows_before}")

    cleaned_df = raw_df.na.drop()

    total_rows_after = cleaned_df.count()

    print(f"Total rows after: {total_rows_after}")

    columns_to_drop = [
        "ArrTime",
        "ActualElapsedTime",
        "AirTime",
        "TaxiIn",
        "Diverted",
        "CarrierDelay",
        "WeatherDelay",
        "NASDelay",
        "SecurityDelay",
        "LateAircraftDelay",
        "Year"
    ]

    filtered_df = raw_df.select([col for col in raw_df.columns if col not in columns_to_drop])
    filtered_df.printSchema()
    return filtered_df


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
        "delimiter": ",",
    }

    raw_df = spark.read.options(**csv_options).csv(file_path)
    # raw_df = spark.read.csv(file_path)
    print(raw_df.show(5))

    cleaned_df = delete(raw_df)
    # cleaned_df = analysis_file(cleaned_df)

    print("Schema of the DataFrame:")
    cleaned_df.printSchema()

    return cleaned_df
