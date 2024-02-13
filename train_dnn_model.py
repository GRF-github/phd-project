import numpy as np

from models.nn.SkDnn import SkDnn
from models.preprocessor.Preprocessors import FgpPreprocessor
from train.param_search import param_search
from train.param_search import create_study


def create_dnn(fgp_cols, binary_cols):
    return SkDnn(use_col_indices=fgp_cols, binary_col_indices=binary_cols, transform_output=True)


def tune_and_fit(alvadesc_data, param_search_config, features):
    """
    features: should be one of "fingerprints", "descriptors" or "all"
    """
    print(f"Starting tune_and_fit with data with dim ({alvadesc_data.X.shape[0]},{alvadesc_data.X.shape[1]})")
    print("Preprocessing...")
    preprocessor = None
    if features == "fingerprints":
        print("Training fingerprints")
        preprocessor = FgpPreprocessor(
            storage=param_search_config.storage,
            study_prefix=f'preproc-{param_search_config.study_prefix}',
            fgp_cols=alvadesc_data.fgp_cols,
            n_trials=param_search_config.n_trials,
            search_cv=param_search_config.param_search_cv
        )

    elif features == "descriptors":
        pass # raise NotImplemented("descriptors not implemented")
    elif features == "all":
        pass # raise NotImplemented("all not implemented")
    else:
        pass # raise ValueError('features: should be one of "fingerprints", "descriptors" or "all"')

    X_train = preprocessor.fit_transform(alvadesc_data.X, alvadesc_data.y)

    print("Creating DNN")
    all_cols = np.arange(X_train.shape[1])
    dnn = SkDnn(use_col_indices=all_cols, binary_col_indices=all_cols[:-1], transform_output=True)

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
    dnn.set_params(**best_params)
    dnn.fit(X_train, alvadesc_data.y)

    return preprocessor, dnn

