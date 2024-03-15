import inspect

import cupy as cp
from xgboost import XGBRegressor

from models.base.PipelineWrapper import RTRegressor

class CudaXGBRegressor(XGBRegressor):
    def fit(self, X, y):
        if self.device == "cuda":
            X = cp.array(X)
            y = cp.array(y)
        else:
            raise ValueError("Cuda XGBRegressor does not support any device other than cuda")
        return super().fit(X, y)


    def predict(self, X):
        if self.device == "cuda":
            X = cp.array(X)
        else:
            raise ValueError("Cuda XGBRegressor does not support any device other than cuda")
        y = super().predict(X)
        return cp.asnumpy(y)



class SelectiveXGBRegressor(RTRegressor):
    def __init__(self, n_estimators=500, max_depth=3, learning_rate=0.1, gamma=1.8,
                 min_child_weight=0.18, subsample=0.95, reg_alpha=0.07, reg_lambda=3.6, colsample_bytree=0.45,
                 colsample_bylevel=0.48, colsample_bynode=0.93, tree_method='hist', verbosity=1,
                 use_col_indices='all', binary_col_indices=None, var_p=0, transform_output=False):
        super().__init__(use_col_indices, binary_col_indices, var_p, transform_output)
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        for arg, val in values.items():
            setattr(self, arg, val)

    def _init_regressor(self):
        return CudaXGBRegressor(**self._rt_regressor_params(), n_jobs=1, booster="gbtree", device="cuda")

