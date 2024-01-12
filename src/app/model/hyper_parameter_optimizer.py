from pyspark.ml.regression import DecisionTreeRegressor, RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

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

