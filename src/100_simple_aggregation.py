"""Extract simple aggregation features

    Reference: https://www.kaggle.com/gpreda/lanl-earthquake-eda-and-prediction
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import competition as cc
from common import stop_watch

TRAIN_CSV_DIRECTORY_PATH = cc.INPUT_PATH / sys.argv[1]
TRAIN_CSV_LIST = list(TRAIN_CSV_DIRECTORY_PATH.glob('**/*.csv'))


@stop_watch
def extract_features(csv_list, feature_dir_path):
    df = pd.DataFrame()
    Path.mkdir(feature_dir_path, exist_ok=True, parents=True)
    for index, each_csv in enumerate(tqdm(sorted(csv_list))):
        seg = pd.read_csv(each_csv, dtype=cc.DTYPES)
        seg_id = each_csv.split("/")[-1].split(".")[0]
        df.loc[index, "seg_id"] = seg_id
        xc = pd.Series(seg['acoustic_data'].values)

        # basic aggregation
        df.loc[index, "mean"] = xc.mean()
        df.loc[index, "std"] = xc.std()
        df.loc[index, "max"] = xc.max()
        df.loc[index, "min"] = xc.min()
        df.loc[index, 'sum'] = xc.sum()
        df.loc[index, 'mad'] = xc.mad()
        df.loc[index, 'kurtosis'] = xc.kurtosis()
        df.loc[index, 'skew'] = xc.skew()
        df.loc[index, 'median'] = xc.median()
        df.loc[index, 'mean_change_rate'] = np.mean(np.nonzero((np.diff(xc) / xc[:-1]))[0])

        # abs aggregation
        df.loc[index, 'abs_mean'] = np.abs(xc).mean()
        df.loc[index, 'abs_std'] = np.abs(xc).std()
        df.loc[index, 'abs_max'] = np.abs(xc).max()
        df.loc[index, 'abs_min'] = np.abs(xc).min()
        df.loc[index, 'abs_sum'] = np.abs(xc).sum()
        df.loc[index, 'abs_mad'] = np.abs(xc).mad()
        df.loc[index, 'abs_kurtosis'] = np.abs(xc).kurtosis()
        df.loc[index, 'abs_skew'] = np.abs(xc).skew()
        df.loc[index, 'abs_median'] = np.abs(xc).median()
        df.loc[index, 'mean_change_abs'] = np.mean(np.diff(xc))
        df.loc[index, 'max_to_min'] = xc.max() / np.abs(xc.min())
        df.loc[index, 'max_to_min_diff'] = xc.max() - np.abs(xc.min())
        df.loc[index, 'count_big'] = len(xc[np.abs(xc) > 500])

    print("Aggregation output is belows:")
    print(df.head(3))
    df.to_csv(feature_dir_path / "{}.csv".format(cc.PREF), index=False)


if __name__ == "__main__":
    train_csv_path = cc.FEATURE_PATH / "{}".format(sys.argv[1])
    train_csv_l = [str(item) for item in TRAIN_CSV_LIST]
    extract_features(train_csv_l, train_csv_path)
    test_csv_path = cc.FEATURE_PATH / "test"
    test_csv_l = [str(item) for item in cc.TEST_CSV_LIST]
    extract_features(test_csv_l, test_csv_path)
