"""Debug train data"""
import pandas as pd
import feather
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pathlib import Path
import competition as cc


pd.set_option("display.max_columns", 180)

seg = cc.INPUT_PATH / "001" / "seg_0.f"
xc = feather.read_dataframe(str(seg))
print(xc["acoustic_data"].mean())

assert False

train_feature = cc.FEATURE_PATH / "001" / "100.f"
xc = pd.Series(seg['acoustic_data'].values)


train_feature = cc.FEATURE_PATH / "001" / "100.f"
seg = feather.read_dataframe(str(train_feature))
print(seg.head(10))

feature_list = ["100"]
feature_files = sorted([str(Path(cc.FEATURE_PATH / "001" / "{}.f".format(f))) for f in feature_list])
target_df = feather.read_dataframe(str(Path(cc.FEATURE_PATH  / "001" / "target.f")))
train_X = pd.concat([feather.read_dataframe(f) for f in tqdm(feature_files, mininterval=30)], axis=1)

scaler = StandardScaler()
scaler.fit(train_X)
scaled_train_X = pd.DataFrame(scaler.transform(train_X), columns=train_X.columns)
print(scaled_train_X.head(10))
