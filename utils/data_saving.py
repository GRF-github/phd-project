import pickle
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_absolute_percentage_error


def _evaluate_all_estimators(blender, X_test, y_test, metrics, fold_number):
    """Evaluate all estimators in blender on the test set"""
    iteration_results = []
    for estimator_name, estimator in blender._fitted_estimators + [('Blender', blender)]:
        estimator_results = {k: metric(y_test, estimator.predict(X_test)) for k, metric in metrics.items()}
        estimator_results['estimator'] = estimator_name
        estimator_results['fold'] = fold_number
        iteration_results.append(estimator_results)
    return pd.DataFrame(iteration_results)

def save_preprocessor_and_blender(preprocessor, blender, test_split_X, test_split_y, results, fold):
    with open(f"./results/blender-preprocessor-{fold}.pkl", "wb") as f:
        pickle.dump(preprocessor, f)
    with open(f"./results/blender-{fold}.pkl", "wb") as f:
        pickle.dump(blender, f)

    X_test = preprocessor.transform(test_split_X)
    metrics = {'mae': mean_absolute_error, 'medae': median_absolute_error, 'mape': mean_absolute_percentage_error}
    results.append(
        _evaluate_all_estimators(blender, X_test, test_split_y, metrics, fold)