import sys
import numpy as np
import pandas as pd
import feather
from pathlib import Path
from tqdm import tqdm
import competition as cc
from common import stop_watch

PREF = sys.argv[0].split("/")[-1].split("_")[0]
TRAIN_FEATHER_DIRECTORY_PATH = cc.INPUT_PATH / sys.argv[1]
TRAIN_FEATHER_LIST = list(TRAIN_FEATHER_DIRECTORY_PATH.glob('**/*.f'))


@stop_watch
def extract_features(feather_list, feature_dir_path):
    df = pd.DataFrame()
    Path.mkdir(feature_dir_path, exist_ok=True, parents=True)
    for index, each_feather in enumerate(tqdm(feather_list)):
        seg = feather.read_dataframe(str(each_feather))
        xc = pd.Series(seg['acoustic_data'].values)

        df.loc[index, "mean"] = xc.mean()
        df.loc[index, "std"] = xc.std()
        df.loc[index, "max"] = xc.max()
        df.loc[index, "min"] = xc.min()

    print("Aggregation output is belows:")
    print(df.head(3))
    df.to_feather(str(feature_dir_path / "{}.f".format(PREF)))


def main():
    print("Extracting features from train dataset {} ...".format(sys.argv[1]))
    train_feature_path = cc.FEATURE_PATH / "{}".format(sys.argv[1])
    extract_features(TRAIN_FEATHER_LIST, train_feature_path)
    print("Extracting features from test dataset ...")
    test_feature_path = cc.FEATURE_PATH / "test"
    extract_features(cc.TEST_FEATHER_LIST, test_feature_path)


if __name__ == "__main__":
    main()
