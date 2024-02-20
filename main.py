import pandas as pd
import pickle
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import StratifiedKFold

from utils.data_loading import get_my_data
from utils.stratification import stratify_y
from train_dnn_model import tune_and_fit

from sklearn.model_selection import RepeatedKFold
from collections import namedtuple

# Parameters
features_list = ["fingerprints", "descriptors", "all"]
SMOKE = False
################

if SMOKE:
    print("Running smoke test...")
    amount_of_data = 2000
    number_of_folds = 2
    number_of_trials = 1
    param_search_folds = 2
    database = "sqlite:///./results/smokeDatabaseYouCanDeleteMe.db"
else:
    amount_of_data = "All"
    number_of_folds = 5
    number_of_trials = 100
    param_search_folds = 5
    database = "sqlite:///./results/cv.db"


if __name__=="__main__":
    # Load data
    print("Loading data")
    X, y, desc_cols, fgp_cols = get_my_data(common_cols=['unique_id', 'correct_ccs_avg'])
    if amount_of_data != "All":
        X = X[:amount_of_data]
        y = y[:amount_of_data]


    results = []
    for features in features_list:
        ParamSearchConfig = namedtuple('ParamSearchConfig', ['storage', 'study_prefix', 'param_search_cv', 'n_trials'])
        base_prefix = f"{features}-nnet"
        param_search_config = ParamSearchConfig(
                storage=database,
                study_prefix=base_prefix,
                param_search_cv=RepeatedKFold(n_splits=param_search_folds, n_repeats=1, random_state=42),
                n_trials=number_of_trials
        )
        cross_validation = StratifiedKFold(n_splits=number_of_folds, shuffle=True, random_state=42)

        for fold, (train_index, test_index) in enumerate(cross_validation.split(X, stratify_y(y))):
            param_search_config = param_search_config._replace(
                study_prefix=base_prefix + f"-fold-{fold}"
            )
            train_split_X = X[train_index]  # Unused
            train_split_y = y[train_index]  # Unused
            test_split_X = X[test_index]
            test_split_y = y[test_index]
            # Prepare for XGB
            preprocessor, dnn = (
                tune_and_fit(X, y, desc_cols, fgp_cols, param_search_config=param_search_config, features=features)
            )

            print("Saving preprocessor and DNN")
            with open(f"./results/preprocessor-{features}-{fold}.pkl", "wb") as f:
                pickle.dump(preprocessor, f)
            with open(f"./results/dnn-{features}-{fold}.pkl", "wb") as f:
                pickle.dump(dnn, f)

            X_test = preprocessor.transform(test_split_X)
            metrics = {'mae': mean_absolute_error, 'medae': median_absolute_error, 'mape': mean_absolute_percentage_error}
            dnn_results = {k: metric(test_split_y, dnn.predict(X_test)) for k, metric in metrics.items()}
            dnn_results['fold'] = fold
            dnn_results['features'] = features
            pd.DataFrame([dnn_results])
            results.append(pd.DataFrame([dnn_results]))

            # Save all intermediate results
            print(f"Saving intermediate results:")
            intermediate_results = pd.concat(results, axis=0)
            intermediate_results.to_csv(f"./results/partial_results{len(results)}.txt", index=False)

    # Print and save final results
    results = pd.concat(results, axis=0)
    print(f"Saving final results")
    print(results)
    results.to_csv(f"./results/cross_validation_results.txt", index=False)

