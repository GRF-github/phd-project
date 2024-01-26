import argparse
import pickle
from collections import namedtuple

import numpy as np
from sklearn.model_selection import RepeatedKFold

from data import AlvadescDataset
from models.nn.SkDnn import SkDnn
from models.preprocessor.Preprocessors import FgpPreprocessor
from train.param_search import param_search
from train.param_search import create_study

import os


def handle_saving_dir(save_to_dir, error_msg=None):
    if not os.path.exists(save_to_dir):
        os.mkdir(save_to_dir)
    elif not os.path.isdir(save_to_dir):
        if error_msg is None:
            error_msg = f"{save_to_dir} should be a directory"
        raise ValueError(error_msg)


ParamSearchConfig = namedtuple('ParamSearchConfig', ['storage', 'study_prefix', 'param_search_cv', 'n_trials'])


def create_dnn(fgp_cols, binary_cols):
    return SkDnn(use_col_indices=fgp_cols, binary_col_indices=binary_cols, transform_output=True)


def tune_and_fit(alvadesc_data, param_search_config):
    print(f"Starting tune_and_fit with data with dim ({alvadesc_data.X.shape[0]},{alvadesc_data.X.shape[1]})")
    print("Preprocessing...")
    preprocessor = FgpPreprocessor(
        storage=param_search_config.storage,
        study_prefix=f'preproc-{param_search_config.study_prefix}',
        fgp_cols=alvadesc_data.fgp_cols,
        n_trials=param_search_config.n_trials,
        search_cv=param_search_config.param_search_cv
    )
    X_train = preprocessor.fit_transform(alvadesc_data.X, alvadesc_data.y)

    print("Creating DNN")
    all_cols = np.arange(X_train.shape[1])
    dnn = create_dnn(fgp_cols=all_cols, binary_cols=all_cols[:-1])

    print("Param search")
    study = create_study("dnn", param_search_config.study_prefix, param_search_config.storage)
    best_params = param_search(
        dnn,
        X_train, alvadesc_data.y,
        cv=param_search_config.param_search_cv,
        study=study,
        n_trials=param_search_config.n_trials,
        keep_going=False
    )
    print("Training")
    dnn = create_dnn(fgp_cols=all_cols, binary_cols=all_cols[:-1])
    dnn.set_params(**best_params)
    dnn.fit(X_train, alvadesc_data.y)

    return preprocessor, dnn


def create_train_parser(default_storage, default_study):
    parser = argparse.ArgumentParser(description="Train DNN")
    parser.add_argument('--storage', type=str, default=default_storage,
                        help='SQLITE DB for storing the results of param search (e.g.: sqlite:///train.db')
    parser.add_argument('--study', type=str, default=default_study,
                        help='Study name to identify param search results withing the DB')
    parser.add_argument('--param_search_folds', type=int, default=5, help='Number of folds to be used in param search')
    parser.add_argument('--trials', type=int, default=10, help='Number of trials in param search')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random state for reproducibility or reusing param search results')
    parser.add_argument('--save_to', type=str, default='.', help='folder where to save the preprocessor and regressor models')
    return parser


def load_data_and_configs(args, download_directory):
    alvadesc_data = AlvadescDataset(download_directory)

    param_search_config = ParamSearchConfig(
        storage=args.storage,
        study_prefix=args.study,
        param_search_cv=RepeatedKFold(n_splits=args.param_search_folds, n_repeats=1, random_state=args.random_state),
        n_trials=args.trials
    )
    return alvadesc_data, param_search_config


if __name__ == '__main__':
    import os
    parser = create_train_parser(default_storage="sqlite:///dnn.db", default_study="dnn")
    args = parser.parse_args()
    handle_saving_dir(args.save_to)

    alvadesc_data, param_search_config = load_data_and_configs(args, download_directory="rt_data")
    print(args)

    preprocessor, dnn = (
        tune_and_fit(alvadesc_data, param_search_config=param_search_config)
    )

    print("Saving preprocessor and DNN")
    with open(os.path.join(args.save_to, "preprocessor.pkl"), "wb") as f:
        pickle.dump(preprocessor, f)
    with open(os.path.join(args.save_to, "dnn.pkl"), "wb") as f:
        pickle.dump(dnn, f)
