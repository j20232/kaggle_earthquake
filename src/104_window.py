"""Extract signal processing features related to window functions

    Reference: https://www.kaggle.com/gpreda/lanl-earthquake-eda-and-prediction
"""

import sys
import numpy as np
import pandas as pd
import feather
from pathlib import Path
from scipy.signal import convolve, hann, hilbert
from tqdm import tqdm
import competition as cc
from common import stop_watch

TRAIN_FEATHER_DIRECTORY_PATH = cc.INPUT_PATH / sys.argv[1]
TRAIN_FEATHER_LIST = list(TRAIN_FEATHER_DIRECTORY_PATH.glob('**/*.f'))


@stop_watch
def extract_features(feather_list, feature_dir_path):
    df = pd.DataFrame()
    Path.mkdir(feature_dir_path, exist_ok=True, parents=True)
    for index, each_feather in enumerate(tqdm(sorted(feather_list))):
        seg = feather.read_dataframe(str(each_feather))
        xc = pd.Series(seg['acoustic_data'].values)

        df.loc[index, 'Hilbert_mean'] = np.abs(hilbert(xc)).mean()
        df.loc[index, 'Hann_window_mean'] = (convolve(xc, hann(150), mode='same') / sum(hann(150))).mean()

        for windows in [10, 100, 1000]:
            x_roll_std = xc.rolling(windows).std().dropna().values
            x_roll_mean = xc.rolling(windows).mean().dropna().values

            df.loc[index, 'ave_roll_std_' + str(windows)] = x_roll_std.mean()
            df.loc[index, 'std_roll_std_' + str(windows)] = x_roll_std.std()
            df.loc[index, 'max_roll_std_' + str(windows)] = x_roll_std.max()
            df.loc[index, 'min_roll_std_' + str(windows)] = x_roll_std.min()
            df.loc[index, 'q01_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.01)
            df.loc[index, 'q05_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.05)
            df.loc[index, 'q95_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.95)
            df.loc[index, 'q99_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.99)
            df.loc[index, 'av_change_abs_roll_std_' + str(windows)] = np.mean(np.diff(x_roll_std))
            df.loc[index, 'av_change_rate_roll_std_' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_std) / x_roll_std[:-1]))[0])
            df.loc[index, 'abs_max_roll_std_' + str(windows)] = np.abs(x_roll_std).max()

            df.loc[index, 'ave_roll_mean_' + str(windows)] = x_roll_mean.mean()
            df.loc[index, 'std_roll_mean_' + str(windows)] = x_roll_mean.std()
            df.loc[index, 'max_roll_mean_' + str(windows)] = x_roll_mean.max()
            df.loc[index, 'min_roll_mean_' + str(windows)] = x_roll_mean.min()
            df.loc[index, 'q01_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.01)
            df.loc[index, 'q05_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.05)
            df.loc[index, 'q95_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.95)
            df.loc[index, 'q99_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.99)
            df.loc[index, 'av_change_abs_roll_mean_' + str(windows)] = np.mean(np.diff(x_roll_mean))
            df.loc[index, 'av_change_rate_roll_mean_' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_mean) / x_roll_mean[:-1]))[0])
            df.loc[index, 'abs_max_roll_mean_' + str(windows)] = np.abs(x_roll_mean).max()

    print("Aggregation output is belows:")
    print(df.head(3))
    df.to_feather(str(feature_dir_path / "{}.f".format(cc.PREF)))


if __name__ == "__main__":
    train_feature_path = cc.FEATURE_PATH / "{}".format(sys.argv[1])
    train_feather_l = [str(item) for item in TRAIN_FEATHER_LIST]
    extract_features(train_feather_l, train_feature_path)
    test_feature_path = cc.FEATURE_PATH / "test"
    test_feather_l = [str(item) for item in cc.TEST_FEATHER_LIST]
    extract_features(test_feather_l, test_feature_path)
