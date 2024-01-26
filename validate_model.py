"""Validate regressors on SMRT dataset

This script allows the user to train and validate a set of different regressors on the SMRT dataset
using cross-validation. Since training includes Bayesian optimization of hyperparameters,
nested cross-validation is actually used.

The models include:
* Gaussian process with deep kernels (deep kernel learning)
* Deep Neural network trained with warm restarts and Stochastic Weight averaging (SWA).
* Gradient Boosting Machines:
    * XGBoost
    * LightGBM
    * A set of CatBoost models assigning different weights to retained and non-retained molecules
* An ensemble of the previous models based on a Random Forest meta-regressors.

With the exception of the CatBoost models, the models are trained using three types of
features: 1) fingerprints, 2) descriptors and 3) fingerprints + descriptors. The
fingerprints and descriptors were obtained using the Alvadesc software.

This script permits the user to specify command line options. Use
$ python validate_model.py --help
to see the options.
"""
import argparse
import os
import tempfile

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import StratifiedKFold


from train_dnn_model import load_data_and_configs
from train_dnn_model import tune_and_fit


from sklearn.model_selection import train_test_split


def stratify_y(y, n_strats=6):
    ps = np.linspace(0, 1, n_strats)
    # make sure last is 1, to avoid rounding issues
    ps[-1] = 1
    quantiles = np.quantile(y, ps)
    cuts = pd.cut(y, quantiles, include_lowest=True)
    codes, _ = cuts.factorize()
    return codes


def stratified_train_test_split(X, y, *, test_size, n_strats=6):
    return train_test_split(X, y, test_size=test_size, stratify=stratify_y(y, n_strats))

def create_base_parser(default_storage, default_study, description=""):
    """Command line parser for both training and validating all models"""
    parser = argparse.ArgumentParser(description=description)

    def restricted_float(x):
        try:
            x = float(x)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{x} not a floating-point literal")
        if x < 0.0 or x > 1.0:
            raise argparse.ArgumentTypeError(f"{x} not in range [0.0, 1.0]")
        return x

    parser.add_argument('--storage', type=str, default=default_storage,
                        help='SQLITE DB for storing the results of param search (e.g.: sqlite:///train.db')
    parser.add_argument('--study', type=str, default=default_study,
                        help='Study name to identify param search results withing the DB')
    parser.add_argument('--train_size', type=restricted_float, default=restricted_float(0.8),
                        help="Percentage of the training set to train the base classifiers. The remainder is used to "
                             "train the meta-classifier")
    parser.add_argument('--param_search_folds', type=int, default=5, help='Number of folds to be used in param search')
    parser.add_argument('--trials', type=int, default=500, help='Number of trials in param search')
    parser.add_argument('--smoke_test', action='store_true',
                        help='Use small model and subsample training data for quick testing. '
                             'param_search_folds and trials are also overridden')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random state for reproducibility or reusing param search results')
    return parser

def create_cv_parser(default_storage, default_study, description):
    """Command line parser for validating all models"""
    parser = create_base_parser(default_storage=default_storage, default_study=default_study, description=description)
    parser.add_argument("--cv_folds", type=int, default=10, help="Number of folds to be used for CV")
    parser.add_argument("--csv_output", type=str, default=os.path.join(tempfile.gettempdir(), "cv_results.csv"),
                        help="CSV file to store the CV results")
    return parser


def evaluate_dnn(dnn, X_test, y_test, metrics, fold_number):
    dnn_results = {k: metric(y_test, dnn.predict(X_test)) for k, metric in metrics.items()}
    dnn_results['fold'] = fold_number
    return pd.DataFrame([dnn_results])


if __name__ == '__main__':
    parser = create_cv_parser(
        description="Cross-validate Blender", default_storage="sqlite:///cv.db", default_study="cv"
    )
    args = parser.parse_args()

    alvadesc_data, param_search_config = load_data_and_configs(args, download_directory="rt_data")

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=args.random_state + 500)
    strats = stratify_y(alvadesc_data.y)

    metrics = {'mae': mean_absolute_error, 'medae': median_absolute_error, 'mape': mean_absolute_percentage_error}

    base_study_name = param_search_config.study_prefix

    results = []
    for fold, (train_index, test_index) in enumerate(cv.split(alvadesc_data.X, strats)):
        param_search_config = param_search_config._replace(study_prefix=base_study_name + f"-fold-{fold}")
        alvadesc_train = alvadesc_data[train_index]
        alvadesc_test = alvadesc_data[test_index]

        preprocessor, dnn = (tune_and_fit(alvadesc_data, param_search_config=param_search_config))
        X_test = preprocessor.transform(alvadesc_test.X)

        results.append(evaluate_dnn(dnn, X_test, alvadesc_test.y, metrics, fold))

        # Temporary code
        print(f"Saving intermediate results results:")
        intermediate_results = pd.concat(results, axis=0)
        intermediate_results.to_csv(f"./results/partial_results{len(results)}.txt", index=False)

    results = pd.concat(results, axis=0)

    print(f"Saving results to {args.csv_output}")
    print(results)
    results.to_csv(args.csv_output, index=False)
