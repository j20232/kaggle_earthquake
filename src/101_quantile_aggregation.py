"""Extract aggregation features using specified areas

    Reference: https://www.kaggle.com/gpreda/lanl-earthquake-eda-and-prediction
"""

import sys
import numpy as np
import pandas as pd
import feather
from pathlib import Path
from scipy import stats
from tqdm import tqdm
import competition as cc
from common import stop_watch

TRAIN_FEATHER_DIRECTORY_PATH = cc.INPUT_PATH / sys.argv[1]
TRAIN_FEATHER_LIST = list(TRAIN_FEATHER_DIRECTORY_PATH.glob('**/*.f'))


@stop_watch
def extract_features(feather_list, feature_dir_path):
    df = pd.DataFrame()
    Path.mkdir(feature_dir_path, exist_ok=True, parents=True)
    for index, each_feather in enumerate(tqdm(feather_list)):
        seg = feather.read_dataframe(str(each_feather))
        xc = pd.Series(seg['acoustic_data'].values)

        # quantile
        df.loc[index, 'q999'] = np.quantile(xc, 0.999)
        df.loc[index, 'q99'] = np.quantile(xc, 0.99)
        df.loc[index, 'q95'] = np.quantile(xc, 0.95)
        df.loc[index, 'q05'] = np.quantile(xc, 0.05)
        df.loc[index, 'q01'] = np.quantile(xc, 0.01)
        df.loc[index, 'q001'] = np.quantile(xc, 0.001)
        df.loc[index, 'iqr'] = np.subtract(*np.percentile(xc, [75, 25]))
        df.loc[index, 'ave10'] = stats.trim_mean(xc, 0.1)

        # abs quantile
        df.loc[index, 'abs_q999'] = np.quantile(np.abs(xc), 0.999)
        df.loc[index, 'abs_q99'] = np.quantile(np.abs(xc), 0.99)
        df.loc[index, 'abs_q95'] = np.quantile(np.abs(xc), 0.95)
        df.loc[index, 'abs_q05'] = np.quantile(np.abs(xc), 0.05)
        df.loc[index, 'abs_q01'] = np.quantile(np.abs(xc), 0.01)
        df.loc[index, 'abs_q001'] = np.quantile(np.abs(xc), 0.001)
        df.loc[index, 'abs_iqr'] = np.subtract(*np.percentile(np.abs(xc), [75, 25]))
        df.loc[index, 'abs_ave10'] = stats.trim_mean(np.abs(xc), 0.1)

        # mean
        df.loc[index, 'avg_first_50000'] = xc[:50000].mean()
        df.loc[index, 'avg_last_50000'] = xc[-50000:].mean()
        df.loc[index, 'avg_first_10000'] = xc[:10000].mean()
        df.loc[index, 'avg_last_10000'] = xc[-10000:].mean()

        # std
        df.loc[index, 'std_first_50000'] = xc[:50000].std()
        df.loc[index, 'std_last_50000'] = xc[-50000:].std()
        df.loc[index, 'std_first_10000'] = xc[:10000].std()
        df.loc[index, 'std_last_10000'] = xc[-10000:].std()

        # max
        df.loc[index, 'max_first_50000'] = xc[:50000].max()
        df.loc[index, 'max_last_50000'] = xc[-50000:].max()
        df.loc[index, 'max_first_10000'] = xc[:10000].max()
        df.loc[index, 'max_last_10000'] = xc[-10000:].max()

        # min
        df.loc[index, 'min_first_50000'] = xc[:50000].min()
        df.loc[index, 'min_last_50000'] = xc[-50000:].min()
        df.loc[index, 'min_first_10000'] = xc[:10000].min()
        df.loc[index, 'min_last_10000'] = xc[-10000:].min()

        # sum
        df.loc[index, 'sum_first_50000'] = xc[:50000].sum()
        df.loc[index, 'sum_last_50000'] = xc[-50000:].sum()
        df.loc[index, 'sum_first_10000'] = xc[:10000].sum()
        df.loc[index, 'sum_last_10000'] = xc[-10000:].sum()

        # mad
        df.loc[index, 'mad_first_50000'] = xc[:50000].mad()
        df.loc[index, 'mad_last_50000'] = xc[-50000:].mad()
        df.loc[index, 'mad_first_10000'] = xc[:10000].mad()
        df.loc[index, 'mad_last_10000'] = xc[-10000:].mad()

        # kurtosis
        df.loc[index, 'kurtosis_first_50000'] = xc[:50000].kurtosis()
        df.loc[index, 'kurtosis_last_50000'] = xc[-50000:].kurtosis()
        df.loc[index, 'kurtosis_first_10000'] = xc[:10000].kurtosis()
        df.loc[index, 'kurtosis_last_10000'] = xc[-10000:].kurtosis()

        # skew
        df.loc[index, 'skew_first_50000'] = xc[:50000].skew()
        df.loc[index, 'skew_last_50000'] = xc[-50000:].skew()
        df.loc[index, 'skew_first_10000'] = xc[:10000].skew()
        df.loc[index, 'skew_last_10000'] = xc[-10000:].skew()

        # median
        df.loc[index, 'median_first_50000'] = xc[:50000].median()
        df.loc[index, 'median_last_50000'] = xc[-50000:].median()
        df.loc[index, 'median_first_10000'] = xc[:10000].median()
        df.loc[index, 'median_last_10000'] = xc[-10000:].median()

        # rate
        df.loc[index, 'mean_change_rate_first_50000'] = np.mean(np.nonzero((np.diff(xc[:50000]) / xc[:50000][:-1]))[0])
        df.loc[index, 'mean_change_rate_last_50000'] = np.mean(np.nonzero((np.diff(xc[-50000:]) / xc[-50000:][:-1]))[0])
        df.loc[index, 'mean_change_rate_first_10000'] = np.mean(np.nonzero((np.diff(xc[:10000]) / xc[:10000][:-1]))[0])
        df.loc[index, 'mean_change_rate_last_10000'] = np.mean(np.nonzero((np.diff(xc[-10000:]) / xc[-10000:][:-1]))[0])

    print("Aggregation output is belows:")
    print(df.head(3))
    df.to_feather(str(feature_dir_path / "{}.f".format(cc.PREF)))


def main():
    train_feature_path = cc.FEATURE_PATH / "{}".format(sys.argv[1])
    extract_features(TRAIN_FEATHER_LIST, train_feature_path)
    test_feature_path = cc.FEATURE_PATH / "test"
    extract_features(cc.TEST_FEATHER_LIST, test_feature_path)


if __name__ == "__main__":
    main()
