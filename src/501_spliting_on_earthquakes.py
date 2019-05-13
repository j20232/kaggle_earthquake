"""Spliting on earthquakes using a config file

    Reference: https://www.kaggle.com/c/LANL-Earthquake-Prediction/discussion/83746#latest-526630
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import competition as cc
from common import stop_watch


@stop_watch
def create_validation():
    config_file = list(cc.CONFIG_PATH.glob(sys.argv[1] + "*.json"))[0]
    with config_file.open() as f:
        params = json.load(f)
    params = params["Validation"]
    if params["Version"] != cc.PREF:
        assert False
    feature_files = sorted([str(Path(cc.FEATURE_PATH / params["Data"] / "{}.csv".format(f))) for f in params["Use"].keys()])

    # Read train X
    train_X = None
    for feature_file in tqdm(feature_files):
        feature_version = feature_file.split("/")[-1].split(".")[0]
        feature_list = params["Use"][feature_version]
        if train_X is None:
            if len(feature_list) == 0:
                train_X = pd.read_csv(feature_file)
            else:
                train_X = pd.read_csv(feature_file, usecols=feature_list)
        else:
            train_X = pd.merge(train_X, pd.read_csv(feature_file, usecols=feature_list), how="inner", on="seg_id")
    
    del train_X["seg_id"]
    scaled_train_X = fit_with_scaler(train_X, params)

    # Read train y
    train_y = pd.read_csv(cc.FEATURE_PATH / params["Data"] / "target.csv")
    quake_list = np.where(np.diff(train_y["time_to_failure"]) > 5)[0]
    train_y["ind"] = 0
    train_y["ind"].iloc[quake_list] = 1
    train_y["quake_ind"] = train_y["ind"].cumsum()
    train_y["fold_id"] = (train_y["quake_ind"] % params["Folds"])
    train_y.drop(["ind", "quake_ind"], axis=1, inplace=True)
    scaled_train_X["fold_id"] = train_y["fold_id"].values

    # train
    for fold_ in range(params["Folds"]):
        fold_dir = cc.VALIDATION_PATH / sys.argv[1] / "fold{}".format(fold_)
        Path.mkdir(fold_dir, exist_ok=True, parents=True)
        X_tr = scaled_train_X.query("fold_id!={}".format(fold_))
        X_val = scaled_train_X.query("fold_id=={}".format(fold_))
        y_tr = train_y.query("fold_id!={}".format(fold_))
        y_val = train_y.query("fold_id=={}".format(fold_))
        del X_tr["fold_id"], X_val["fold_id"], y_tr["fold_id"], y_val["fold_id"]

        X_tr.reset_index(inplace=True)
        X_val.reset_index(inplace=True)
        y_tr.reset_index(inplace=True)
        y_val.reset_index(inplace=True)

        X_tr.to_csv(fold_dir / "X_tr.csv", index=False)
        X_val.to_csv(fold_dir / "X_val.csv", index=False)
        y_tr.to_csv(fold_dir / "y_tr.csv", index=False)
        y_val.to_csv(fold_dir / "y_val.csv", index=False)

    # Read test
    feature_files = sorted([str(Path(cc.FEATURE_PATH / "test" / "{}.csv".format(f))) for f in params["Use"].keys()])
    test_X = None
    for feature_file in tqdm(feature_files):
        feature_version = feature_file.split("/")[-1].split(".")[0]
        feature_list = params["Use"][feature_version]
        if test_X is None:
            if len(feature_list) == 0:
                test_X = pd.read_csv(feature_file)
            else:
                test_X = pd.read_csv(feature_file, usecols=feature_list)
        else:
            test_X = pd.merge(test_X, pd.read_csv(feature_file, usecols=feature_list), how="inner", on="seg_id")
    del test_X["seg_id"]
    scaled_test_X = fit_with_scaler(test_X, params)
    scaled_test_X.to_csv(cc.VALIDATION_PATH / sys.argv[1] / "test.csv", index=False)
    print("Test shape: ", scaled_test_X.shape)


def fit_with_scaler(df, params):
    if params["Scaler"] == "StandardScaler":
        scaler = StandardScaler()
    elif params["Scaler"] == "MinMaxScaler":
        scaler = MinMaxScaler()
    else:
        assert False
    df.dropna(inplace=True)
    scaler.fit(df)
    scaled_df = pd.DataFrame(scaler.transform(df), columns=df.columns)
    return scaled_df


if __name__ == "__main__":
    create_validation()
