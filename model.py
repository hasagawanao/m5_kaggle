import os
import gc
import warnings

import pandas as pd
from pandas.plotting import register_matplotlib_converters
import numpy as np
import lightgbm as lgb

#ハイパーパラメータチューニング自動化ライブラリ
import optuna
#LightGBM用Stepwise Tuningに必要
from optuna.integration import lightgbm_tuner


X_train=pd.read_csv("X_train.csv")
y_train=pd.read_csv('y_train.csv')
#X_test=read_csv('X_test.csv')

dt_col = "date"

class CustomTimeSeriesSplitter:
    def __init__(self, n_splits=5, train_days=80, test_days=20, dt_col="date"):
        self.n_splits = n_splits
        self.train_days = train_days
        self.test_days = test_days
        self.dt_col = dt_col

    def split(self, X, y=None, groups=None):
        sec = (X[self.dt_col] - X[self.dt_col][0]).dt.total_seconds()
        duration = sec.max()

        DAYS_TO_SEC = 3600 * 24

        train_sec = self.train_days * DAYS_TO_SEC
        test_sec = self.test_days * DAYS_TO_SEC
        total_sec = test_sec + train_sec

        if self.n_splits == 1:
            train_start = duration - total_sec
            train_end = train_start + train_sec

            train_mask = (sec >= train_start) & (sec < train_end)
            test_mask = sec >= train_end

            yield sec[train_mask].index.values, sec[test_mask].index.values

        else:
            # step = (duration - total_sec) / (self.n_splits - 1)
            step = 7 * DAYS_TO_SEC

            for idx in range(self.n_splits):
                # train_start = idx * step
                shift = (self.n_splits - (idx + 1)) * step
                train_start = duration - total_sec - shift
                train_end = train_start + train_sec
                test_end = train_end + test_sec

                train_mask = (sec >= train_start) & (sec < train_end)

                if idx == self.n_splits - 1:
                    test_mask = sec >= train_end
                else:
                    test_mask = (sec >= train_end) & (sec < test_end)

                yield sec[train_mask].index.values, sec[test_mask].index.values

    def get_n_splits(self):
        return self.n_splits

cv_params = {
    "n_splits": 1,
    "train_days": 365 * 2,
    "test_days": 7,
    "dt_col": dt_col,
}
cv = CustomTimeSeriesSplitter(**cv_params)

def train_lgb(bst_params, fit_params, X, y, cv, drop_when_train=None):
    models = []

    if drop_when_train is None:
        drop_when_train = []

    for idx_fold, (idx_trn, idx_val) in enumerate(cv.split(X, y)):
        print(f"\n---------- Fold: ({idx_fold + 1} / {cv.get_n_splits()}) ----------\n")

        X_trn, X_val = X.iloc[idx_trn], X.iloc[idx_val]
        y_trn, y_val = y.iloc[idx_trn], y.iloc[idx_val]
        train_set = lgb.Dataset(X_trn.drop(drop_when_train, axis=1), label=y_trn)
        val_set = lgb.Dataset(X_val.drop(drop_when_train, axis=1), label=y_val)
        
        best_params, tuning_history = dict(), list()
        print("start")
        model = lightgbm_tuner.train(
            bst_params,
            train_set,
            valid_sets=[train_set, val_set],
            valid_names=["train", "valid"],
            **fit_params,
            #fobj = custom_asymmetric_train, 
            #feval = custom_asymmetric_valid,
            best_params=best_params,
            tuning_history=tuning_history
        )
        models.append(model)
        print(best_params)
        print(tuning_history)

        del idx_trn, idx_val, X_trn, X_val, y_trn, y_val
        gc.collect()

    return models

bst_params = {
        'boosting_type': 'gbdt',
        'objective': 'poisson',
        "metric":"rmse",
        'n_jobs': -1,}
    
fit_params = {
    "num_boost_round": 10000,
    "early_stopping_rounds":50,
    "verbose_eval": 50,
}


models = train_lgb(
    bst_params, fit_params, X_train, y_train, cv, drop_when_train=[dt_col]
)

del X_train, y_train
gc.collect()
print("Done")
