"""Predict labels with lightgbm models"""
import os
import sys
import json
import pandas as pd
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
    config_file = list(cc.CONFIG_PATH.glob(sys.argv[1] + "*.json"))[0]
    with config_file.open() as f:
        params = json.load(f)
    params = params["Training"]

    preds = None
    predict_df = None
    test_csv_path = Path(cc.VALIDATION_PATH / sys.argv[1] / "test.csv")
    test_X = pd.read_csv(test_csv_path)
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
    sample_df = pd.read_csv(cc.SAMPLE_SUBMISSION_CSV_PATH)
    predict_df = pd.DataFrame(predict_df)
    predict_df["seg_id"] = sample_df["seg_id"]
    predict_df["time_to_failure"] = preds
    del predict_df["index"]
    Path.mkdir(cc.SUBMIT_PATH, exist_ok=True, parents=True)
    predict_df.to_csv(cc.SUBMIT_PATH / "{}.csv".format(sys.argv[1]), index=False)


if __name__ == "__main__":
    predict_with_lightgbm()
