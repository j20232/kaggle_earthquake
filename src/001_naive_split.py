"""Split like belows

    train_1: 0 ~ 149999
    train_2: 150000~299999
    ... and so on.

"""

import sys
import numpy as np
import pandas as pd
import feather
from pathlib import Path
from tqdm import tqdm
import competition as cc
from common import stop_watch

PREF = sys.argv[0].split("/")[-1].split("_")[0]


@stop_watch
def main():
    dataset_path = cc.INPUT_PATH / "train_{}".format(PREF)
    Path.mkdir(dataset_path, exist_ok=True, parents=True)
    train_df = feather.read_dataframe(str(cc.TRAIN_FEATHER_PATH))
    segments = int(np.floor(train_df.shape[0] / cc.DATASET_LENGTH))
    print("Number of segments: ", segments)

    train_X = pd.DataFrame(index=range(segments), dtype=np.int16, columns=["acoustic_data"])
    train_y = pd.DataFrame(index=range(segments), dtype=np.float64, columns=['time_to_failure'])
    for seg_id in tqdm(range(segments)):
        segment_df = train_df.iloc[seg_id * cc.DATASET_LENGTH : seg_id * cc.DATASET_LENGTH + cc.DATASET_LENGTH]
        train_y.loc[seg_id, 'time_to_failure'] = segment_df['time_to_failure'].values[-1]
        train_X["acoustic_data"] = segment_df[["acoustic_data"]].copy()
        train_X.to_feather(str(dataset_path / "seg_{}.f".format(seg_id)))
    train_y.to_feather(str(cc.INPUT_PATH / "train_{}.f".format(PREF)))


if __name__ == "__main__":
    main()
