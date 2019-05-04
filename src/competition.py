import numpy as np
from pathlib import Path

DTYPES = {
    'acoustic_data': np.int16,
    'time_to_failure': np.float64
}

ROOT_PATH = Path(__file__).absolute().parents[1]
INPUT_PATH = ROOT_PATH / "input"
FEATURE_PATH = ROOT_PATH / "data" / "features"

# Train Path
TRAIN_CSV_PATH = INPUT_PATH / "train.csv"
TRAIN_FEATHER_PATH = INPUT_PATH / "train.f"

# Test Path
TEST_CSV_DIRECTORY_PATH = INPUT_PATH / "test"
TEST_CSV_LIST = list(TEST_CSV_DIRECTORY_PATH.glob('**/*.csv'))
TEST_FEATHER_DIRECTORY_PATH = INPUT_PATH / "test_feather"
TEST_FEATHER_LIST = list(TEST_CSV_DIRECTORY_PATH.glob('**/*.f'))

# Sample Submission Path
SAMPLE_SUBMISSION_CSV_PATH = INPUT_PATH / "sample_submission.csv"
SAMPLE_SUBMISSION_FEATHER_PATH = INPUT_PATH / "sample_submission.f"
