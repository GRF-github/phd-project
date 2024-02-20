
import argparse
import pickle
from collections import namedtuple

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold

from models.ensemble.Blender import Blender
from models.gbm.xgboost import SelectiveXGBRegressor
from models.nn.SkDnn import SkDnn
from models.preprocessor.Preprocessors import Preprocessor
from train.param_search import param_search

BlenderConfig = namedtuple('BlenderConfig', ['train_size', 'n_strats', 'random_state'])
ParamSearchConfig = namedtuple('ParamSearchConfig', ['storage', 'study_prefix', 'param_search_cv', 'n_trials'])


def create_blender(desc_cols, fgp_cols, binary_cols, blender_config):
    """Create a blender model with the specified configuration"""
    estimators = [
        # Deep Neural Nets
        ('full_mlp', SkDnn(use_col_indices='all', binary_col_indices=binary_cols, transform_output=True)),
        ('desc_mlp', SkDnn(use_col_indices=desc_cols, binary_col_indices=binary_cols, transform_output=True)),
        ('fgp_mlp', SkDnn(use_col_indices=fgp_cols, binary_col_indices=binary_cols, transform_output=True)),
        # XGBoost
        ('full_xgb', SelectiveXGBRegressor(use_col_indices='all', binary_col_indices=binary_cols)),
        ('desc_xgb', SelectiveXGBRegressor(use_col_indices=desc_cols, binary_col_indices=binary_cols)),
        ('fgp_xgb', SelectiveXGBRegressor(use_col_indices=fgp_cols, binary_col_indices=binary_cols))
    ]
    return Blender(
        estimators, RandomForestRegressor(), **blender_config._asdict()
    )

# TODO: change the signature of the funciton to X, y, desc_cols, fgp_cols, param_search_config=param_search_config, features=features
def tune_and_fit(X, y, desc_cols, fgp_cols, *, param_search_config, blender_config):
    """Perform hyperparameter search for all models and fit final models using the best configuration."""
    print(f"Starting tune_and_fit with data with dim ({X.shape[0]},{X.shape[1]})")
    print("Preprocessing...")
    preprocessor = Preprocessor(
        desc_cols=desc_cols,
        fgp_cols=fgp_cols
    )
    X_train = preprocessor.fit_transform(X, y)
    features_description = preprocessor.describe_transformed_features()


    print("Creating blender")
    blender = create_blender(features_description['desc_cols'],
                             features_description['fgp_cols'],
                             features_description['binary_cols'],
                             blender_config)

    print("Param search")
    blender = param_search(
        blender,
        X_train, y,
        cv=param_search_config.param_search_cv,
        study=(param_search_config.storage, param_search_config.study_prefix),
        n_trials=param_search_config.n_trials,
        keep_going=False
    )
    print("Training")
    blender.fit(X_train, y)

    return preprocessor, blender


