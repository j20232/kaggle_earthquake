"""Predict labels with lightgbm models"""
import os
import sys
import json
import pandas as pd
import feather
import lightgbm as lgb
from pathlib import Path
import competition as cc
from common import stop_watch, predict_chunk

# For osx
os.environ['KMP_DUPLICATE_LIB_OK'] = "True"


@stop_watch
def predict_with_lightgbm():
    model_directory_path = cc.MODEL_PATH / sys.argv[1]
    model_path_list = sorted(list(model_directory_path.glob("*.model")))
    config_file = list(Path(cc.CONFIG_PATH / sys.argv[1].split("_")[0]).glob(sys.argv[1] + "*.json"))[0]
    with config_file.open() as f:
        params = json.load(f)
    preds = None
    predict_df = None
    test_feather_path = Path(cc.VALIDATION_PATH / params["Validation"] / "test.f")
    test_X = feather.read_dataframe(str(test_feather_path))
    for fold, model_path in enumerate(model_path_list):
        print("=== [Predict] fold{} starts!! ===".format(fold))
        model = lgb.Booster(model_file=str(model_path))
        if predict_df is None:
            predict_df = test_X["index"]
            test_X = test_X.set_index("index")
        if preds is None:
            preds = predict_chunk(model, test_X) / len(model_path_list)
        else:
            preds += predict_chunk(model, test_X) / len(model_path_list)
    predict_df = pd.DataFrame(predict_df)
    predict_df["time_to_failure"] = preds
    Path.mkdir(cc.SUBMIT_PATH, exist_ok=True, parents=True)
    predict_df.to_csv(cc.SUBMIT_PATH / "{}.csv".format(sys.argv[1]), index=False)


if __name__ == "__main__":
    predict_with_lightgbm()
