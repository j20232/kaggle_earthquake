"""Train a model with lightgbm"""
import os
import sys
import json
import pandas as pd
import lightgbm as lgb
from pathlib import Path
import competition as cc
from common import stop_watch

# For osx
os.environ['KMP_DUPLICATE_LIB_OK'] = "True"


@stop_watch
def train_with_lightgbm():
    config_file = list(cc.CONFIG_PATH.glob(sys.argv[1] + "*.json"))[0]
    with config_file.open() as f:
        params = json.load(f)
    params = params["Training"]
    if params["Version"] != cc.PREF:
        assert False
    validity = None
    feature_importance = pd.DataFrame()
    fold_dir_list = sorted(list(Path(cc.VALIDATION_PATH / sys.argv[1]).glob("fold*")))

    for fold, fold_dir in enumerate(fold_dir_list):
        print("=== [Train] fold{} starts!! ===".format(fold))
        X_tr = pd.read_csv(fold_dir / "X_tr.csv")
        X_val = pd.read_csv(fold_dir / "X_val.csv")
        y_tr = pd.read_csv(fold_dir / "y_tr.csv")
        y_val = pd.read_csv(fold_dir / "y_val.csv")
        X_tr.set_index("index", inplace=True)
        X_val.set_index("index", inplace=True)
        y_tr.set_index("index", inplace=True)
        y_val.set_index("index", inplace=True)
        y_tr = y_tr["time_to_failure"]
        y_val = y_val["time_to_failure"]
        train_dataset = lgb.Dataset(X_tr, y_tr)
        valid_dataset = lgb.Dataset(X_val, y_val)

        if validity is None:
            validity = pd.DataFrame()
            validity["time_to_failure"] = pd.concat([y_tr, y_val])
            validity["Predict"] = 0
            feature_importance["feature"] = X_tr.columns
            feature_importance["median"] = 0

        model = lgb.train(params["trn_params"],
                          train_set=train_dataset,
                          num_boost_round=params["mtd_params"]["num_boost_round"],
                          valid_sets=[train_dataset, valid_dataset],
                          verbose_eval=params["mtd_params"]["verbose_eval"],
                          early_stopping_rounds=params["mtd_params"]["early_stopping_rounds"])
        validity.loc[validity.index.isin(X_val.index), "Predict"] = model.predict(X_val, num_iteration=model.best_iteration)
        feature_importance["fold{}".format(fold)] = model.feature_importance(importance_type="gain")
        cc.save_model(model, fold)
        cc.save_feature_importance(feature_importance)
    cc.save_oof(validity)


if __name__ == "__main__":
    train_with_lightgbm()
