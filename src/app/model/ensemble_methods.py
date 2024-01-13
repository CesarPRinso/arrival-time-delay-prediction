# Training the models of:
#  - Decision Tree
#  - Random Forest
#  - AdaBoost

from src.app.model.hyper_parameter_optimizer import CVreg_tree, CVrandom_forest, CVadaboost
from src.app.model.model_helpers import reg_evaluator, plot_feature_importances, best_hyperparams
import time


def train_decision_tree(Adf_train, Adf_test, feature_cols, spark):
    # Step 3: Train a Decision Tree model with Cross validation for tuning the hyper-parameters
    # ------------------------------------------------------------------------------------------
    start_time = time.time()
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
    # plot_feature_importances(dtr_CVmodel, feature_cols, plot_title='Regression Decision Tree Feature Importances')

    # Obtain the best hyperparameters
    print("Best Hyper-Parameters of Decision Tree")
    dtr_param_names = ["maxDepth", "maxBins", "minInstancesPerNode", "minInfoGain"]
    best_hyperparams(dtr_CVmodel, dtr_param_names)

    elapsed_time = time.time() - start_time
    print(f"Execution time for first model: {elapsed_time} seconds")


def train_random_forest(Adf_train, Adf_test, feature_cols, spark):
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


def train_adaboost(Adf_train, Adf_test, feature_cols, spark):
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
