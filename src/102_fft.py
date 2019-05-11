"""Extract fft features

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
        xc = pd.Series(seg['acoustic_data'].values)
        zc = np.fft.fft(xc)
        realFFT = np.real(zc)
        imagFFT = np.imag(zc)

        # FFT: Real
        df.loc[index, 'Rmean'] = realFFT.mean()
        df.loc[index, 'Rstd'] = realFFT.std()
        df.loc[index, 'Rmax'] = realFFT.max()
        df.loc[index, 'Rmin'] = realFFT.min()

        # FFT: Imaginary
        df.loc[index, 'Imean'] = imagFFT.mean()
        df.loc[index, 'Istd'] = imagFFT.std()
        df.loc[index, 'Imax'] = imagFFT.max()
        df.loc[index, 'Imin'] = imagFFT.min()

        # FFT: Real (Specified area)
        df.loc[index, 'Rmean_last_5000'] = realFFT[-5000:].mean()
        df.loc[index, 'Rmean_last_15000'] = realFFT[-15000:].mean()
        df.loc[index, 'Rstd_last_5000'] = realFFT[-5000:].std()
        df.loc[index, 'Rstd_last_15000'] = realFFT[-15000:].std()
        df.loc[index, 'Rmax_last_5000'] = realFFT[-5000:].max()
        df.loc[index, 'Rmax_last_15000'] = realFFT[-15000:].max()
        df.loc[index, 'Rmin_last_5000'] = realFFT[-5000:].min()
        df.loc[index, 'Rmin_last_15000'] = realFFT[-15000:].min()

        # FFT: Imaginary (Specified area)
        df.loc[index, 'Imean_last_5000'] = imagFFT[-5000:].mean()
        df.loc[index, 'Imean_last_15000'] = imagFFT[-15000:].mean()
        df.loc[index, 'Istd_last_5000'] = imagFFT[-5000:].std()
        df.loc[index, 'Istd_last_15000'] = imagFFT[-15000:].std()
        df.loc[index, 'Imax_last_5000'] = imagFFT[-5000:].max()
        df.loc[index, 'Imax_last_15000'] = imagFFT[-15000:].max()
        df.loc[index, 'Imin_last_5000'] = imagFFT[-5000:].min()
        df.loc[index, 'Imin_last_15000'] = imagFFT[-15000:].min()

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
