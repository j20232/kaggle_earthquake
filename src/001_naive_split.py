"""Split like belows

    train_1: 0 ~ 149999
    train_2: 150000~299999
    ... and so on.

"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import competition as cc
from common import stop_watch


@stop_watch
def main():
    dataset_path = cc.INPUT_PATH / "{}".format(cc.PREF)
    Path.mkdir(dataset_path, exist_ok=True, parents=True)
    print("Reading the train csv file...")
    train_df = pd.read_csv(cc.TRAIN_CSV_PATH, dtype=cc.DTYPES)
    segments = int(np.floor(train_df.shape[0] / cc.DATASET_LENGTH))
    print("Number of segments: ", segments)

    train_X = pd.DataFrame(index=range(segments), dtype=np.int16, columns=["acoustic_data"])
    train_y = pd.DataFrame(index=range(segments), dtype=np.float64, columns=['time_to_failure'])
    for seg_id in tqdm(range(segments)):
        # If you want to change how to split train data, change the following description
        segment_df = train_df.iloc[seg_id * cc.DATASET_LENGTH : seg_id * cc.DATASET_LENGTH + cc.DATASET_LENGTH]

        train_y.loc[seg_id, 'time_to_failure'] = segment_df['time_to_failure'].values[-1]
        train_X = segment_df[["acoustic_data"]].copy()
        train_X.reset_index(inplace=True)
        del train_X["index"]
        train_X.to_csv(dataset_path / "seg_{:0=4}.csv".format(seg_id), index=False)

    feature_dir_path = cc.FEATURE_PATH / cc.PREF
    Path.mkdir(feature_dir_path, exist_ok=True, parents=True)
    train_y.to_csv(str(feature_dir_path / "target.csv"), index=False)


if __name__ == "__main__":
    main()
