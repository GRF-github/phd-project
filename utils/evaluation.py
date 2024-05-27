import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_absolute_percentage_error


def evaluate_model(blender, preprocessor, test_split_X, test_split_y, test_index, fold):
    X_test = preprocessor.transform(test_split_X)
    metrics = {'mae': mean_absolute_error, 'medae': median_absolute_error, 'mape': mean_absolute_percentage_error}

    # Evaluate all estimators in blender on the test set
    iteration_results = []
    for estimator_name, estimator in blender._fitted_estimators + [('Blender', blender)]:
        estimator_results = {k: metric(test_split_y, estimator.predict(X_test)) for k, metric in metrics.items()}
        estimator_results['estimator'] = estimator_name
        estimator_results['fold'] = fold
        iteration_results.append(estimator_results)

        # 1st column are the indices, 2nd column are the experimental values, 3rd column are the predicted values (plot y against x)
        pred_vs_exp_df = pd.DataFrame()
        pred_vs_exp_df['test_index'] = test_index
        pred_vs_exp_df['test_split_y'] = test_split_y
        pred_vs_exp_df['predicted_values'] = estimator.predict(X_test)
        pred_vs_exp_df.to_csv(f"./results/pred_vs_exp_{estimator_name}.csv", index=False, mode='a', header=False)

    results = pd.DataFrame(iteration_results)

    print()

    # Save all intermediate results
    results.to_csv(f"./results/evaluation_results.txt", index=False, mode='a', header=False)
    print(f"Evaluation results have been saved into ./results/evaluation_results_short.txt")
