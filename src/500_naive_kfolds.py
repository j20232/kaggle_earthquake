"""Naive KFold Validation using a config file"""

import sys
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import KFold
import competition as cc
from common import stop_watch


@stop_watch
def create_validation():
    config_file = list(cc.CONFIG_PATH.glob(sys.argv[1] + "*.json"))[0]
    with config_file.open() as f:
        params = json.load(f)
    params = params["Validation"]
    feature_files = sorted([str(Path(cc.FEATURE_PATH / params["Data"] / "{}.csv".format(f))) for f in params["Features"]])

    # Read train X
    train_X = pd.concat([
        pd.read_csv(f) for f in tqdm(feature_files, mininterval=30)], axis=1)
    train_X.drop(params["Drop"], axis=1, inplace=True)
    scaled_train_X = fit_with_scaler(train_X, params)
    print("scaled_train_X.shape: {}".format(scaled_train_X.shape))

    # Read train y
    train_y = pd.read_csv(cc.FEATURE_PATH / params["Data"] / "target.csv")
    print("train_y.shape: {}".format(train_y.shape))
    assert scaled_train_X.shape[0] == train_y.shape[0]

    # Read test
    feature_files = sorted([str(Path(cc.FEATURE_PATH / "test" / "{}.csv".format(f))) for f in params["Features"]])
    test_X = pd.concat([
        pd.read_csv(f) for f in tqdm(feature_files, mininterval=30)], axis=1)
    test_X.drop(params["Drop"], axis=1, inplace=True)
    scaled_test_X = fit_with_scaler(test_X, params)
    print("scaled_test_X.shape: {}".format(scaled_test_X.shape))
    assert scaled_train_X.shape[1] == scaled_test_X.shape[1]

    kf = KFold(n_splits=params["Folds"], shuffle=True, random_state=params["Seed"])
    for fold_, (trn_idx, val_idx) in enumerate(kf.split(scaled_train_X, train_y.values)):
        fold_dir = cc.VALIDATION_PATH / sys.argv[1] / "fold{}".format(fold_)
        Path.mkdir(fold_dir, exist_ok=True, parents=True)
        X_tr, X_val = scaled_train_X.iloc[trn_idx], scaled_train_X.iloc[val_idx]
        y_tr, y_val = train_y.iloc[trn_idx], train_y.iloc[val_idx]

        X_tr.reset_index(inplace=True)
        X_val.reset_index(inplace=True)
        y_tr.reset_index(inplace=True)
        y_val.reset_index(inplace=True)

        X_tr.to_csv(fold_dir / "X_tr.csv", index=False)
        X_val.to_csv(fold_dir / "X_val.csv", index=False)
        y_tr.to_csv(fold_dir / "y_tr.csv", index=False)
        y_val.to_csv(fold_dir / "y_val.csv", index=False)
    scaled_test_X.reset_index(inplace=True)
    scaled_test_X.to_csv(cc.VALIDATION_PATH / sys.argv[1] / "test.csv")


def fit_with_scaler(df, params):
    if params["Scaler"] == "StandardScaler":
        scaler = StandardScaler()
    elif params["Scaler"] == "MinMaxScaler":
        scaler = MinMaxScaler()
    else:
        assert False
    scaler.fit(df)
    scaled_df = pd.DataFrame(scaler.transform(df), columns=df.columns)
    return scaled_df


if __name__ == "__main__":
    create_validation()
