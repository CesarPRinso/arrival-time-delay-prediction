# Training the models of: 
#  - Decision Tree
#  - Random Forest
#  - AdaBoost

from src.app.model.ensemble_methods import train_decision_tree, train_random_forest, train_adaboost
from src.app.model.hyper_parameter_optimizer import CVreg_tree
from src.app.model.model_helpers import split_train_test, vector_assembler, prepare_data
from pyspark import SparkContext


# ----------------------------------------------------------------------------------
#                               MAIN FUNCTION                                     |
# ----------------------------------------------------------------------------------

def parallelize_tasks(transformations, spark):
    try:
        # Paralelizar las tareas y obtener resultados
        results_rdd = spark.sparkContext.parallelize(transformations)
        results_collected = results_rdd.map(lambda x: (x[0], x[1](*x[2])))
        results_collected = results_collected.collect()

        # Procesar los resultados si es necesario
        for task_name, result in results_collected:
            print(f"{task_name} completed with result: {result}")

    except Exception as e:
        # Manejar cualquier excepción y registrar el error
        print(f"Error during parallel execution: {e}")
        return None


def modelTuning(df, spark):
    # Step 1 and Step 2:
    Adf_train, Adf_test, feature_cols = prepare_data(df)
    train_decision_tree(Adf_train, Adf_test, feature_cols, spark)
    train_random_forest(Adf_train, Adf_test, feature_cols, spark)
    train_adaboost(Adf_train, Adf_test, feature_cols, spark)
