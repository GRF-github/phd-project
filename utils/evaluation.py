import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_absolute_percentage_error


def evaluate_model(blender, preprocessor, test_split_X, test_split_y, fold):
    X_test = preprocessor.transform(test_split_X)
    metrics = {'mae': mean_absolute_error, 'medae': median_absolute_error, 'mape': mean_absolute_percentage_error}

    # Evaluate all estimators in blender on the test set
    iteration_results = []
    for estimator_name, estimator in blender._fitted_estimators + [('Blender', blender)]:
        estimator_results = {k: metric(test_split_y, estimator.predict(X_test)) for k, metric in metrics.items()}
        estimator_results['estimator'] = estimator_name
        estimator_results['fold'] = fold
        iteration_results.append(estimator_results)
    results = pd.DataFrame(iteration_results)

    # Save all intermediate results
    results.to_csv(f"./results/evaluation_results.txt", index=False, mode='a', header=False)
    print(f"Evaluation results have been saved into ./results/evaluation_results_short.txt")
