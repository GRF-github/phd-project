from collections import namedtuple

from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import StratifiedKFold

from train_model import tune_and_fit
from utils.data_loading import get_my_data
from utils.data_saving import save_preprocessor_and_blender
from utils.evaluation import evaluate_model
from utils.stratification import stratify_y

# Parameters
is_smoke_test = False
################

if is_smoke_test:
    print("Running smoke test...")
    number_of_folds = 2
    number_of_trials = 1
    param_search_folds = 2
    database = "sqlite:///./results/smokeDatabaseYouCanDeleteMe.db"
else:
    number_of_folds = 5
    number_of_trials = 15
    param_search_folds = 5
    database = "sqlite:///./results/cv.db"


if __name__ == "__main__":
    # Load data
    print("Loading data")
    X, y, desc_cols, fgp_cols = get_my_data(common_cols=['unique_id', 'correct_ccs_avg'], is_smoke_test=is_smoke_test)

    BlenderConfig = namedtuple('BlenderConfig', ['train_size', 'n_strats', 'random_state'])
    blender_config = BlenderConfig(train_size=0.8, n_strats=8, random_state=3674)

    ParamSearchConfig = namedtuple('ParamSearchConfig', ['storage', 'study_prefix', 'param_search_cv', 'n_trials'])
    param_search_config = ParamSearchConfig(
            storage=database,
            study_prefix="blender",
            param_search_cv=RepeatedKFold(n_splits=param_search_folds, n_repeats=1, random_state=42),
            n_trials=number_of_trials
    )

    cross_validation = StratifiedKFold(n_splits=number_of_folds, shuffle=True, random_state=435)

    for fold, (train_index, test_index) in enumerate(cross_validation.split(X, stratify_y(y))):
        param_search_config = param_search_config._replace(
            study_prefix=f"cv-fold-{fold}"
        )
        train_split_X = X[train_index]
        train_split_y = y[train_index]
        test_split_X = X[test_index]
        test_split_y = y[test_index]

        preprocessor, blender = (
            tune_and_fit(train_split_X, train_split_y, desc_cols, fgp_cols,
                         param_search_config=param_search_config, blender_config=blender_config)
        )

        save_preprocessor_and_blender(preprocessor, blender, fold)
        evaluate_model(blender, preprocessor, test_split_X, test_split_y, fold)
