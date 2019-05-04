"""Create feather files from input files"""

import gc
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import competition as cc
from common import stop_watch


@stop_watch
def train():
    train_df = pd.read_csv(cc.TRAIN_CSV_PATH, dtype=cc.DTYPES)
    train_df.to_feather(str(cc.TRAIN_FEATHER_PATH))
    Path.mkdir(cc.FEATURE_PATH, exist_ok=True, parents=True)
    train_df[["time_to_failure"]].to_feather(str(cc.FEATURE_PATH / "target.f"))
    del train_df
    gc.collect()


@stop_watch
def test():
    Path.mkdir(cc.TEST_FEATHER_DIRECTORY_PATH, exist_ok=True, parents=True)
    if len(cc.TEST_CSV_LIST) != 2624:
        assert False
    for test_csv in tqdm(cc.TEST_CSV_LIST):
        file_name = test_csv.name.split(".")[0] + ".f"
        test_df = pd.read_csv(test_csv, dtype=cc.DTYPES)
        test_df.to_feather(str(cc.TEST_FEATHER_DIRECTORY_PATH / file_name))
        del test_df
        gc.collect()


@stop_watch
def sample():
    sample_df = pd.read_csv(cc.SAMPLE_SUBMISSION_CSV_PATH, dtype=cc.DTYPES)
    sample_df.to_feather(str(cc.SAMPLE_SUBMISSION_FEATHER_PATH))
    del sample_df
    gc.collect()


if __name__ == "__main__":
    gc.enable()
    train()
    test()
    sample()
