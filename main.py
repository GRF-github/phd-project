import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import StratifiedKFold

from data import AlvadescDataset
from train_dnn_model import tune_and_fit

from sklearn.model_selection import RepeatedKFold
from collections import namedtuple

# Parameters
amount_of_data = "All"  # default all        minimum 2000
number_of_folds = 5  # default 10           minimum 2
number_of_trials = 100  # default 100         minimum 1
param_search_folds = 2  # default 5         minimum 2
param_search_trials = 1  # default 3        minimum 1

# Load data
print("Loading data")
data = AlvadescDataset("rt_data")
if amount_of_data != "All":
    data = data[:amount_of_data]


def stratify_y(y, n_strats=6):
    ps = np.linspace(0, 1, n_strats)
    # make sure last is 1, to avoid rounding issues
    ps[-1] = 1
    quantiles = np.quantile(y, ps)
    cuts = pd.cut(y, quantiles, include_lowest=True)
    codes, _ = cuts.factorize()
    return codes


ParamSearchConfig = namedtuple('ParamSearchConfig', ['storage', 'study_prefix', 'param_search_cv', 'n_trials'])
param_search_config = ParamSearchConfig(
        storage="sqlite:///./results/cv.db",
        study_prefix="study",
        param_search_cv=RepeatedKFold(n_splits=param_search_folds, n_repeats=param_search_trials, random_state=42),
        n_trials=number_of_trials)

cross_validation = StratifiedKFold(n_splits=number_of_folds, shuffle=True, random_state=42)

results = []
for fold, (train_index, test_index) in enumerate(cross_validation.split(data.X, stratify_y(data.y))):
    param_search_config = param_search_config._replace(study_prefix="study" + f"-fold-{fold}")

    train_split = data[train_index]
    test_split = data[test_index]
    preprocessor, dnn = (tune_and_fit(data, param_search_config=param_search_config))

    print("Saving preprocessor and DNN")
    with open(f"./results/preprocessor{fold}.pkl", "wb") as f:
        pickle.dump(preprocessor, f)
    with open(f"./results/dnn{fold}.pkl", "wb") as f:
        pickle.dump(dnn, f)

    X_test = preprocessor.transform(test_split.X)
    metrics = {'mae': mean_absolute_error, 'medae': median_absolute_error, 'mape': mean_absolute_percentage_error}
    dnn_results = {k: metric(test_split.y, dnn.predict(X_test)) for k, metric in metrics.items()}
    dnn_results['fold'] = fold
    pd.DataFrame([dnn_results])
    results.append(pd.DataFrame([dnn_results]))

    # Save all intermediate results
    print(f"Saving intermediate results results:")
    intermediate_results = pd.concat(results, axis=0)
    intermediate_results.to_csv(f"./results/partial_results{len(results)}.txt", index=False)

# Print and save final results
results = pd.concat(results, axis=0)
print(f"Saving final results")
print(results)
results.to_csv(f"./results/cross_validation_results.txt", index=False)
