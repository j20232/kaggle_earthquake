# kaggle_earthquake
For kaggle

がんばるぞい

# Index List

- `000` ~ `099`: Preparation for data
    - `000`: Create feather files
    - `001` ~ `030`: Create train datasets
    - `031` ~ `060`: Data augmentation
    - `061` ~ `099`: Validation
- `100` ~ `599`: Feature extraction
- `600` ~ `799`: Training
- `800` ~ `899`: Prediction
- `900` ~ `999`: Visualization

# Installation

1. Put input files to `./input`
2. Create feather files

```
python src/000_to_feather.py
```

3. Create train datasets

```
python src/001_naive_split.py
```

4. Extract features by specifying the index of the input datasets

```
python src/100_simple_aggregation.py 001
```

5. Train a model using an input config file

```
TODO
```

6. Predict test files using a input model

```
TODO
```
