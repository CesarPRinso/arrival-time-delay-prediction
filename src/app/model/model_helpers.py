import os

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
import matplotlib.pyplot as plt


def prepare_data(df, spark):
    # Step 1: Split the DataFrame into into 70% for train and 30% for test
    # ----------------------------------------------------------------------
    df_train, df_test = split_train_test(df, train_ratio=0.7, spark=spark)
    # Check the sizes of the resulting DataFrames
    # print("Training set size:", df_train.count())
    # print("Test set size:", df_test.count())

    # List of features excluding the target variable 'ArrDelay'
    feature_cols = [col for col in df.columns if col != 'ArrDelay']

    # Step 2: Assemble train and test sets
    # -------------------------------------
    Adf_train = vector_assembler(df_train, feature_cols)
    Adf_test = vector_assembler(df_test, feature_cols)
    # Show the new dataframes
    # Adf_train.show(truncate=False)
    # Adf_test.show(truncate=False)

    return Adf_train, Adf_test, feature_cols

# Function for spliting the dataframe into 70% for train and 30% for test
def split_train_test(dataframe, train_ratio, spark) :
    # Randomly split into train and test
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.abspath(os.path.join(current_dir, '../../'))
    parquet_dir = os.path.join(src_dir, 'parquet')
    parquet_path = os.path.join(parquet_dir, 'combined_data.parquet')
    dataframe = spark.read.parquet(parquet_path)

    splits = dataframe.randomSplit([train_ratio, 1.0 - train_ratio], seed=7)

    # Obtain the dataframes od train and test
    df_train = splits[0]
    df_test = splits[1]

    return df_train, df_test


# Function for the vector assembler and creating a vector for all the individual features
def vector_assembler(dataframe, feature_cols):
    # Vector Assembler excluding 'ArrDelay'
    assembler = VectorAssembler().setInputCols(feature_cols).setOutputCol('features')

    # Apply the assembler to the DataFrame
    a_df = assembler.transform(dataframe)
    b_df = a_df.select("features", a_df.ArrDelay.alias('label'))

    return b_df


# Function for evaluating a regression model
def reg_evaluator(test_dt, spark):
    evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction")
    metrics = ["r2", "mse", "rmse", "mae"]
    metric_values = [evaluator.evaluate(test_dt, {evaluator.metricName: metric}) for metric in metrics]

    # Obtain a dataframe with those results
    metric_df = spark.createDataFrame(zip(metrics, metric_values), ["Metric", "Value"])
    return metric_df


# Function for obtaining the hyper-parameters from the best model obtained
def best_hyperparams(model, param_names):
    for param_name in param_names:
        param = model.getOrDefault(param_name)
        print(f"Best {param_name}: {param}")


# Function for getting the feature importances of a model
def plot_feature_importances(model, feature_cols, plot_title):
    var_importances = model.featureImportances

    feature_imp_list = [(feature_cols[i], var_importances[i]) for i in range(len(feature_cols))]
    sorted_imp = sorted(feature_imp_list, key=lambda x: x[1], reverse=True)

    for feature, importance in sorted_imp:
        print(f"Feature: {feature}, Importance: {importance}")

    features, importance_values = zip(*sorted_imp)
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(features)), importance_values, align='center')
    plt.yticks(range(len(features)), features)
    plt.xlabel('Importance')
    plt.title(plot_title)
    plt.show()
