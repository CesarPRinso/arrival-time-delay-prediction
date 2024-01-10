# Training the models of: 
#  - Decision Tree
#  - Random Forest
#  - AdaBoost

from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col
from pyspark.ml.regression import DecisionTreeRegressor, RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import matplotlib.pyplot as plt


# Function for spliting the dataframe into 70% for train and 30% for test
def split_train_test(dataframe, train_ratio):
    # Randomly split into train and test
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


# Function for training a decision tree model for regression using k-fold (k=5) cross validation to find the best hyperparameters
def CVreg_tree(df_train):
    dtr = DecisionTreeRegressor()
    # Define a grid of hyperparameters to search through
    hyperparamGrid = (
        ParamGridBuilder()
        .addGrid(dtr.maxDepth, [3, 5, 7, 10])  # default = 5
        .addGrid(dtr.maxBins, [32, 35, 40]) #default = 32
        .addGrid(dtr.minInstancesPerNode, [1, 3, 5, 7])  # default = 1
        .addGrid(dtr.minInfoGain, [0.0, 0.2])  # default = 0.0
        .build()
    )

    # Define an evaluator
    evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")

    # Create a k-fold CrossValidator with k=5
    crossval = CrossValidator(
        estimator=dtr,
        estimatorParamMaps=hyperparamGrid,
        evaluator=evaluator,
        numFolds=5,
        seed=7
    )
    cvModel = crossval.fit(df_train)

    # The best model from cross-validation
    best_model = cvModel.bestModel

    return best_model


# Function for training a random forest model for regression using k-fold (k=5) cross validation to find the best hyperparameters
def CVrandom_forest(df_train):
    rf = RandomForestRegressor()
    # Define a grid of hyperparameters to search
    hyperparamGrid = (
        ParamGridBuilder()
        .addGrid(rf.numTrees, [10, 20, 35])  # default = 20
        .addGrid(rf.maxDepth, [10])  # default = 5, it was chosen the best hyper-parameter from the decision tree
        .addGrid(rf.maxBins, [40])  # default = 32, it was chosen the best hyper-parameter from the decision tree
        .addGrid(rf.subsamplingRate, [0.6, 0.8, 1.0])  # default = 1
        .addGrid(rf.minInstancesPerNode,
                 [5])  # default = 1, it was chosen the best hyper-parameter from the decision tree
        .addGrid(rf.minInfoGain, [0.0])  # default = 0.0, it was chosen the best hyper-parameter from the decision tree
        .build()
    )
    # Evaluator
    evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")

    # K-fold CrossValidator
    crossval = CrossValidator(
        estimator=rf,
        estimatorParamMaps=hyperparamGrid,
        evaluator=evaluator,
        numFolds=5,
        seed=7
    )
    cvModel = crossval.fit(df_train)

    # The best model from cross-validation
    best_model = cvModel.bestModel

    return best_model


# Function for training an AdaBoost model for regression using k-fold (k=5) cross validation to find the best hyperparameters 
def CVadaboost(df_train):
    ab = GBTRegressor(maxDepth=1)  # In AdaBoost each tree is a stump(1 root node and 2 leafs nodes)
    # Define a grid of hyperparameters to search
    hyperparamGrid = (
        ParamGridBuilder()
        .addGrid(ab.maxIter, [10, 20, 35, 50])  # default = 20
        .addGrid(ab.stepSize, [0.1, 0.2, 0.3])  # default = 0.1
        .build()
    )
    # Evaluator
    evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")

    # K-fold CrossValidator
    crossval = CrossValidator(
        estimator=ab,
        estimatorParamMaps=hyperparamGrid,
        evaluator=evaluator,
        numFolds=5,
        seed=7
    )
    cvModel = crossval.fit(df_train)

    # The best model from cross-validation
    best_model = cvModel.bestModel

    return best_model


# ----------------------------------------------------------------------------------
#                               MAIN FUNCTION                                     |
# ----------------------------------------------------------------------------------

def modelTuning(df, spark):
    # Step 1: Split the DataFrame into into 70% for train and 30% for test
    # ----------------------------------------------------------------------
    df_train, df_test = split_train_test(df, train_ratio=0.7)
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

    # Step 3: Train a Decision Tree model with Cross validation for tuning the hyper-parameters
    # ------------------------------------------------------------------------------------------
    dtr_CVmodel = CVreg_tree(Adf_train)

    # Test the model
    test_CVdtr = dtr_CVmodel.transform(Adf_test)
    # test_CVdtr.show(truncate=False)

    # Evaluate the model
    print('Evaluating the Regression Tree Model:')
    CVdtr_evaluation = reg_evaluator(test_CVdtr, spark)
    CVdtr_evaluation.show()

    # Obtain the decision tree
    CVtree_structure = dtr_CVmodel.toDebugString
    print("Regression Tree Model:")
    print(CVtree_structure)
    names_df = spark.createDataFrame([(f'feature {idx}', col) for idx, col in enumerate(feature_cols)],
                                     ["Feature Position", "Feature Name"])
    names_df.show()

    # Get the feature importances
    plot_feature_importances(dtr_CVmodel, feature_cols, plot_title='Regression Decision Tree Feature Importances')

    # Obtain the best hyperparameters
    print("Best Hyper-Parameters of Decision Tree")
    dtr_param_names = ["maxDepth", "maxBins", "minInstancesPerNode", "minInfoGain"]
    best_hyperparams(dtr_CVmodel, dtr_param_names)

    # Step 4: Train a Random Forest model with Cross validation for tuning the hyper-parameters
    # ------------------------------------------------------------------------------------------
    rf_CVmodel = CVrandom_forest(Adf_train)

    # Test the model
    test_CVrf = rf_CVmodel.transform(Adf_test)
    # test_CVrf.show(truncate=False)

    # Evaluate the model
    print('Evaluating the Ranfom Forest Model:')
    CVrf_evaluation = reg_evaluator(test_CVrf, spark)
    CVrf_evaluation.show()

    # Get the feature importances
    plot_feature_importances(rf_CVmodel, feature_cols, plot_title='Random Forest Feature Importances')

    # Obtain the best hyperparameters
    print("Best Hyper-Parameters of Random Forest")
    rf_param_names = ["numTrees", "maxDepth", "maxBins", "subsamplingRate", "minInstancesPerNode", "minInfoGain"]
    best_hyperparams(rf_CVmodel, rf_param_names)

    # Step 5: Train an AdaBoost model with Cross validation for tuning the hyper-parameters
    # --------------------------------------------------------------------------------------
    ab_CVmodel = CVadaboost(Adf_train)

    # Test the model
    test_CVab = ab_CVmodel.transform(Adf_test)
    # test_CVab.show(truncate=False)

    # Evaluate the model
    print('Evaluating the AdaBoost Model:')
    CVab_evaluation = reg_evaluator(test_CVab, spark)
    CVab_evaluation.show()

    # Get the feature importances
    plot_feature_importances(ab_CVmodel, feature_cols, plot_title='AdaBoost Feature Importances')

    # Obtain the best hyperparameters
    print("Best Hyper-Parameters of AdaBoost")
    ab_param_names = ["maxDepth", "maxIter", "stepSize"]
    best_hyperparams(ab_CVmodel, ab_param_names)
