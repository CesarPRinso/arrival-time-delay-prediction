# PREPROCESSING OF raw_df

from pyspark.sql import functions as F
from pyspark.sql.functions import col, sum as spark_sum, mean as _mean
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, MinMaxScaler, PCA, Word2Vec
from pyspark.sql.types import StringType, IntegerType, DoubleType
from pyspark.ml import Pipeline
from functools import reduce
from pyspark.sql.functions import monotonically_increasing_id


def overview(file_path, spark):
    # Read the raw file
    csv_options = {
        "header": True,
        "inferSchema": True,
        "delimiter": ",",
    }
    raw_df = spark.read.options(**csv_options).csv(file_path).repartition(10)

    print("Overview of the original dataset:")
    print("Number of instances: ", raw_df.count())
    print("Number of columns: ", len(raw_df.columns))
    print("Name and type of each variable:")
    raw_df.printSchema()
    raw_df.persist()

    return raw_df


def clean(raw_df):
    # Remove forbidden and unuseful variables
    df_drop = raw_df.drop(
        *['ArrTime', 'ActualElapsedTime', 'AirTime', 'TaxiIn', 'Diverted', 'CarrierDelay', 'WeatherDelay', 'NASDelay',
          'SecurityDelay', 'LateAircraftDelay', 'UniqueCarrier', 'TailNum', 'Cancelled', 'CancellationCode',
          'CRSDepTime'])
    print("Number of columns after removing the forbidden variables: ", len(df_drop.columns))
    return df_drop


def see_null(df):
    # See if the dataframe has null values
    if df.dropna().count() == df.count():
        print("There are no null values in the dataframe.")
    else:
        print("There are null values in the dataframe.")

    # Sum null values in each column
    null_counts = df.select(
        [spark_sum(col(c).isNull().cast("int")).alias(c) for c in df.columns])
    null_counts.show()
    return df


def remove_one(df):
    # Some variables have only one valid value, so we will remove them.
    df_drop_one = df

    for one in df_drop_one.columns:
        if df_drop_one.select(one).distinct().count() == 1:
            print(one)
            df_drop_one = df_drop_one.drop(one)
    print("Number of columns after removing one value variables: ", len(df_drop_one.columns))
    return df_drop_one


def remove_instances(df):
    # Remove instances where target variable is a missing value.
    df_arr_delay = df.filter(df.ArrDelay != "NA")

    print("Number of remaining flights: ", df_arr_delay.count())

    # See the resulting dataframe
    df_arr_delay.printSchema()
    df_arr_delay.count()
    return df_arr_delay


def variable_type(df):
    # Change variable types
    df_rep = df.replace(("NA"), None)

    # We convert some numerical variables into string and string to int
    catCols = [x for (x, dataType) in df_rep.dtypes if dataType == "string"]
    numCols = [x for (x, dataType) in df_rep.dtypes if dataType != "string"]

    cat_type = ["Origin", "Dest"]
    df_cat = df_rep
    for cat in df_cat.columns:
        if cat in cat_type:
            print(cat)
            df_cat = df_cat.withColumn(cat, df_cat[cat].cast(StringType()))

    # We convert some string variables into integer
    for inte in df_cat.columns:
        if inte not in cat_type:
            print(inte)
            df_cat = df_cat.withColumn(inte, df_cat[inte].cast(IntegerType()))

    df_cat.printSchema()
    return df_cat


def count_missing(spark_df, sort=True):
    df = spark_df.select([F.count(F.when(F.isnull(c), c)).alias(c) for (c, c_type) in spark_df.dtypes]).toPandas()
    if len(df) == 0:
        print("There are no any missing values!")
        return None
    if sort:
        return df.rename(index={0: 'count'}).T.sort_values("count", ascending=False)
    return df


def fill_missing(df):
    # If the number of missing values is over 70%, the variable is removed.
    df_before_dropping = df
    list_missing = count_missing(df_before_dropping, sort=False)
    list_missing_array = list_missing.values.tolist()
    list_missing_array = list_missing_array[0]
    list_drop = []

    print("List of columns dropped:")
    for col in range(0, len(df_before_dropping.columns)):
        if list_missing_array[col] > 0.7 * df_before_dropping.count():
            list_drop.append(df_before_dropping.columns[col])
    for i in range(0, len(list_drop)):
        print(list_drop[i])
        df_before_dropping = df_before_dropping.drop(list_drop[i])
    df_dropped_missing = df_before_dropping
    print("Total: ", len(df.columns) - len(df_dropped_missing.columns), 'columns dropped')

    # Fill missing values for the rest of variables
    catCols = [x for (x, dataType) in df_dropped_missing.dtypes if dataType == "string"]
    numCols = [x for (x, dataType) in df_dropped_missing.dtypes if dataType != "string"]
    print("We have categorical variables: ", catCols)
    print("We have numerical variables: ", numCols)
    print("")
    print("These columns are going to be filled as follows:")
    print("- Categorical values: with 0.")  # check how to fill
    print("- Numerical values: with the mean of the column.")
    print("")

    for col in range(0, len(df_dropped_missing.columns)):
        name_col = df_dropped_missing.columns[col]
        if name_col in catCols:  # If categorical
            df_dropped_missing = df_dropped_missing.fillna({name_col: 0})
        else:
            df_stats = df_dropped_missing.select(_mean(name_col).alias('mean')).collect()
            mean = df_stats[0]['mean']
            df_dropped_missing = df_dropped_missing.fillna(mean, subset=[name_col])
            df_dropped_missing = df_dropped_missing.fillna(mean, subset=[name_col])
    clean_df = df_dropped_missing

    # Check if the number of missing values is 0
    print("Check if the missing data has been filled.")
    n_missings = count_missing(clean_df, sort=True)
    print(n_missings)
    return clean_df


def correlation(df, spark):
    intCols = [x for (x, dataType) in df.dtypes if dataType == "int"]
    corr = []
    for i in intCols:
        corr.append(i)
        corr.append(df.stat.corr(i, "ArrDelay"))

    print(corr)
    # correlation_df = spark.createdataframe(corr, ["variable", "correlation"])
    #
    # # convert spark dataframe to pandas
    # correlation_pd = correlation_df.topandas()
    #
    # sns.set(style="whitegrid")
    # plt.figure(figsize=(12, 6))
    # ax = sns.barplot(x="variable", y="correlation", data=correlation_pd)
    # ax.set(title="correlation between variables and arrdelay")
    # plt.xticks(rotation=45, ha='right')
    # plt.show()


def contingency_table(df):
    catCols = [x for (x, dataType) in df.dtypes if dataType == "string"]
    numCols = [x for (x, dataType) in df.dtypes if dataType != "string"]

    selected_columns = [col(column) for column in catCols + numCols]

    contingency_table = df.select(selected_columns).groupBy(catCols).agg({col: "mean" for col in numCols})

    for num_col in numCols:
        new_col_name = f"Mean_{num_col}"
        contingency_table = contingency_table.withColumnRenamed(f"avg({num_col})", new_col_name)

    contingency_table.show()


def one_hot_encode(df):
    print("Transforming categorical variables using one-hot encoding...")
    catCols = [x for (x, dataType) in df.dtypes if dataType == "string"]
    print("Variables to be transformed:", catCols)
    string_indexer = [
        StringIndexer(inputCol=x, outputCol=x + "_StringIndexer", handleInvalid="skip")
        for x in catCols
    ]

    one_hot_encoder = [
        OneHotEncoder(
            inputCols=[f"{x}_StringIndexer" for x in catCols],
            outputCols=[f"{x}_OneHotEncoder" for x in catCols],
        )
    ]

    stages = []
    stages += string_indexer
    stages += one_hot_encoder

    pipeline = Pipeline().setStages(stages)
    model = pipeline.fit(df)
    df_encoded = model.transform(df)

    notStringCols = [x for (x, dataType) in df_encoded.dtypes if ((dataType != "string") and (dataType != "double"))]
    df_encoded_clean = df_encoded.select([col for col in notStringCols])
    print("Let's see how the data set looks like after one-hot encoding:")
    # df_encoded_clean.printSchema()
    return df_encoded_clean


def embeddings_encode(df):
    print("Transforming categorical variables using embeddings...")
    cat_cols = [x for (x, data_type) in df.dtypes if data_type == "string"]
    print("Variables to be transformed:", cat_cols)
    word2vec_models = [
        Word2Vec(vectorSize=5, minCount=1, inputCol=col, outputCol=f"{col}_embedding")
        for col in cat_cols
    ]
    pipeline = Pipeline(stages=word2vec_models)
    model = pipeline.fit(df)
    df_embedded = model.transform(df)

    # Seleccionar solo las columnas deseadas
    selected_cols = [c for c in df.columns if c not in cat_cols]  # Columnas originales
    selected_cols += [f"{col}_embedding" for col in cat_cols]  # Columnas embeddings

    # Seleccionar solo las columnas deseadas en el DataFrame resultante
    df_result = df_embedded.select(selected_cols)

    return df_result


def string_indexer_and_join(df):
    index_col_name = "index"
    df = df.withColumn(index_col_name, F.monotonically_increasing_id())

    indexed_df = df

    for col_name in df.columns:
        if col_name != index_col_name and df.select(col_name).dtypes[0][1] == "string":
            indexer = StringIndexer(inputCol=col_name, outputCol=f"{col_name}_cat")
            indexed_df = indexer.fit(indexed_df).transform(indexed_df)


    for col_name in df.columns:
        if col_name != index_col_name and df.select(col_name).dtypes[0][1] == "string":
            indexed_df = indexed_df.drop(col_name)

    return indexed_df


def pca(df):
    columns_for_pca = [col for col in df.columns if col != 'ArrDelay']

    assembler = VectorAssembler(
        inputCols=columns_for_pca,
        outputCol="features"
    )

    pca = PCA(k=2, inputCol="features", outputCol="pca_features")
    pipeline = Pipeline(stages=[assembler, pca])
    model = pipeline.fit(df)
    df_pca = model.transform(df)
    df_pca = df_pca.select("*", "pca_features", "ArrDelay")

    return df_pca


def preprocess(file_path, spark):
    # Step 1: Read the original dataframe
    df = overview(file_path, spark)

    # Step 2: Clean the dataframe
    df_cleaned = clean(df)

    # Step 3: See if there are null instances
    df_without_null = see_null(df_cleaned)

    # Step 4: Remove variables with one value
    # df_without_one = remove_one(df_without_null)

    # Step 5: Remove instances where target variable is a missing value
    df_without_null_inst = remove_instances(df_without_null)

    # Step 6: Cast variable types
    df_type = variable_type(df_without_null_inst)

    # Step 7: Check missing values and fill them
    df_clean2 = fill_missing(df_type)
    print(df_clean2.show(5))
    df_clean2.printSchema()

    # Step 8: Use one-hot encoding to transform categorical variables
    df_onehot = string_indexer_and_join(df_clean2)
    # print(df_onehot.show(5))
    # df_onehot.printSchema()
    # Convert Spark DataFrame to Pandas DataFrame
    pandas_df_onehot = df_onehot.toPandas()
    # Save Pandas DataFrame as CSV locally
    pandas_df_onehot.to_csv('dataOneHot.csv', index=False)

    # Step 9. PCA analysis
    df_pca = pca(df_onehot)
    print(df_pca.show(5))
    df_pca.printSchema()

    # Correlation analysis
    # correlation(df_onehot,spark)
    # contingency_table(df_clean2)

    return df_pca
