"""Extract signal processing features

    Reference: https://www.kaggle.com/gpreda/lanl-earthquake-eda-and-prediction
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
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
        
        # Regression
        df.loc[index, 'trend'] = add_trend_feature(xc)
        df.loc[index, 'abs_trend'] = add_trend_feature(xc, abs_values=True)

        # classic_sta_lta (the definition is written in this file)
        df.loc[index, 'classic_sta_lta1_mean'] = classic_sta_lta(xc, 500, 10000).mean()
        df.loc[index, 'classic_sta_lta2_mean'] = classic_sta_lta(xc, 5000, 100000).mean()
        df.loc[index, 'classic_sta_lta3_mean'] = classic_sta_lta(xc, 3333, 6666).mean()
        df.loc[index, 'classic_sta_lta4_mean'] = classic_sta_lta(xc, 10000, 25000).mean()

        # moving average
        df.loc[index, 'Moving_average_700_mean'] = xc.rolling(window=700).mean().mean(skipna=True)
        df.loc[index, 'Moving_average_1500_mean'] = xc.rolling(window=1500).mean().mean(skipna=True)
        df.loc[index, 'Moving_average_3000_mean'] = xc.rolling(window=3000).mean().mean(skipna=True)
        df.loc[index, 'Moving_average_6000_mean'] = xc.rolling(window=6000).mean().mean(skipna=True)

        # ema moving average
        ewma = pd.Series.ewm
        df.loc[index, 'exp_Moving_average_300_mean'] = ewma(xc, span=300).mean().mean(skipna=True)
        df.loc[index, 'exp_Moving_average_3000_mean'] = ewma(xc, span=3000).mean().mean(skipna=True)
        df.loc[index, 'exp_Moving_average_30000_mean'] = ewma(xc, span=6000).mean().mean(skipna=True)

        # moving average by correction with std
        no_of_std = 2
        df.loc[index, 'MA_400MA_std_mean'] = xc.rolling(window=400).std().mean()
        df.loc[index, 'MA_400MA_BB_high_mean'] = (df.loc[index, 'Moving_average_700_mean'] + no_of_std * df.loc[index, 'MA_400MA_std_mean']).mean()
        df.loc[index, 'MA_400MA_BB_low_mean'] = (df.loc[index, 'Moving_average_700_mean'] - no_of_std * df.loc[index, 'MA_400MA_std_mean']).mean()
        df.loc[index, 'MA_700MA_std_mean'] = xc.rolling(window=700).std().mean()
        df.loc[index, 'MA_700MA_BB_high_mean'] = (df.loc[index, 'Moving_average_700_mean'] + no_of_std * df.loc[index, 'MA_700MA_std_mean']).mean()
        df.loc[index, 'MA_700MA_BB_low_mean'] = (df.loc[index, 'Moving_average_700_mean'] - no_of_std * df.loc[index, 'MA_700MA_std_mean']).mean()
        df.loc[index, 'MA_1000MA_std_mean'] = xc.rolling(window=1000).std().mean()

    print("Aggregation output is belows:")
    print(df.head(3))
    df.to_csv(feature_dir_path / "{}.csv".format(cc.PREF), index=False)


def add_trend_feature(arr, abs_values=False):
    idx = np.array(range(len(arr)))
    if abs_values:
        arr = np.abs(arr)
    lr = LinearRegression()
    lr.fit(idx.reshape(-1, 1), arr)
    return lr.coef_[0]


def classic_sta_lta(x, length_sta, length_lta):
    sta = np.cumsum(x ** 2)
    # Convert to float
    sta = np.require(sta, dtype=np.float)
    # Copy for LTA
    lta = sta.copy()
    # Compute the STA and the LTA
    sta[length_sta:] = sta[length_sta:] - sta[:-length_sta]
    sta /= length_sta
    lta[length_lta:] = lta[length_lta:] - lta[:-length_lta]
    lta /= length_lta
    # Pad zeros
    sta[:length_lta - 1] = 0
    # Avoid division by zero by setting zero values to tiny float
    dtiny = np.finfo(0.0).tiny
    idx = lta < dtiny
    lta[idx] = dtiny
    return sta / lta


if __name__ == "__main__":
    train_csv_path = cc.FEATURE_PATH / "{}".format(sys.argv[1])
    train_csv_l = [str(item) for item in TRAIN_CSV_LIST]
    extract_features(train_csv_l, train_csv_path)
    test_csv_path = cc.FEATURE_PATH / "test"
    test_csv_l = [str(item) for item in cc.TEST_CSV_LIST]
    extract_features(test_csv_l, test_csv_path)
