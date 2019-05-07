import sys
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error

DTYPES = {
    'acoustic_data': np.int16,
    'time_to_failure': np.float32
}

PREF = sys.argv[0].split("/")[-1].split("_")[0]
ROOT_PATH = Path(__file__).absolute().parents[1]
INPUT_PATH = ROOT_PATH / "input"
CONFIG_PATH = ROOT_PATH / "config"
FEATURE_PATH = ROOT_PATH / "data" / "features"
VALIDATION_PATH = ROOT_PATH / "data" / "validation"
MODEL_PATH = ROOT_PATH / "data" / "model"
OOF_PATH = ROOT_PATH / "data" / "oof"
IMPORTANCE_PATH = ROOT_PATH / "importance"
SUBMIT_PATH = ROOT_PATH / "data" / "submit"

# Train Path
TRAIN_CSV_PATH = INPUT_PATH / "train.csv"
TRAIN_FEATHER_PATH = INPUT_PATH / "train.f"

# Test Path
TEST_CSV_DIRECTORY_PATH = INPUT_PATH / "test"
TEST_CSV_LIST = list(TEST_CSV_DIRECTORY_PATH.glob('**/*.csv'))
TEST_FEATHER_DIRECTORY_PATH = INPUT_PATH / "test_feather"
TEST_FEATHER_LIST = list(TEST_FEATHER_DIRECTORY_PATH.glob('**/*.f'))

# Sample Submission Path
SAMPLE_SUBMISSION_CSV_PATH = INPUT_PATH / "sample_submission.csv"
SAMPLE_SUBMISSION_FEATHER_PATH = INPUT_PATH / "sample_submission.f"

# Const
DATASET_LENGTH = 150000


def save_oof(validity):
    validity = validity.reset_index()
    columns_order = ["index", "time_to_failure", "Predict"]
    validity = validity.sort_values("index").reset_index(drop=True).loc[:, columns_order]
    cv_mae = (mean_absolute_error(validity["time_to_failure"], np.array(validity["Predict"])))
    print("!!!!! CV MAE:{}".format(cv_mae))
    Path.mkdir(OOF_PATH, exist_ok=True, parents=True)
    validity.to_csv(OOF_PATH / "{}.csv".format(sys.argv[1]))


def save_feature_importance(feature_importance):
    feature_importance["median"] = feature_importance.median(axis='columns')
    feature_importance.sort_values("median", ascending=False, inplace=True)
    Path.mkdir(IMPORTANCE_PATH, exist_ok=True, parents=True)
    feature_importance.to_csv(Path(IMPORTANCE_PATH / "{}.csv".format(sys.argv[1])))


def save_model(model, fold):
    model_path = MODEL_PATH / sys.argv[1]
    Path.mkdir(model_path, exist_ok=True, parents=True)
    model.save_model(str(model_path / "fold{}.model".format(fold)))
