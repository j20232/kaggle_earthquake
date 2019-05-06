"""Naive KFold Validation using a config file"""

import sys
import json
import pandas as pd
import feather
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import KFold
import competition as cc
from common import stop_watch


@stop_watch
def create_validation():
    config_file = list(Path(cc.CONFIG_PATH / cc.PREF).glob(sys.argv[1] + "*.json"))[0]
    with config_file.open() as f:
        params = json.load(f)
    feature_files = [str(Path(cc.FEATURE_PATH / params["Data"] / "{}.f".format(f))) for f in params["Features"]]

    # Read train X
    train_X = pd.concat([
        feather.read_dataframe(f) for f in tqdm(feature_files, mininterval=30)], axis=1)
    train_X.drop(params["Drop"], axis=1, inplace=True)
    scaled_train_X = fit_with_scaler(train_X, params)
    print("scaled_train_X.shape: {}".format(scaled_train_X.shape))

    # Read train y
    train_y = feather.read_dataframe(str(Path(cc.FEATURE_PATH / params["Data"] / "target.f")))
    print("train_y.shape: {}".format(train_y.shape))
    assert scaled_train_X.shape[0] == train_y.shape[0]

    # Read test
    feature_files = [str(Path(cc.FEATURE_PATH / "test" / "{}.f".format(f))) for f in params["Features"]]
    test_X = pd.concat([
        feather.read_dataframe(f) for f in tqdm(feature_files, mininterval=30)], axis=1)
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

        X_tr.reset_index(drop=True, inplace=True)
        X_val.reset_index(drop=True, inplace=True)
        y_tr.reset_index(drop=True, inplace=True)
        y_val.reset_index(drop=True, inplace=True)

        X_tr.to_feather(str(fold_dir / "X_tr.f"))
        X_val.to_feather(str(fold_dir / "X_val.f"))
        y_tr.to_feather(str(fold_dir / "y_tr.f"))
        y_val.to_feather(str(fold_dir / "y_val.f"))
    scaled_test_X.to_feather(str(cc.VALIDATION_PATH / sys.argv[1] / "test.f"))


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