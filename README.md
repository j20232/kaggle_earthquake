# kaggle_earthquake
For kaggle

がんばるぞい

# Index List

- `000` ~ `099`: Preparation for data
    - `000`: Create feather files
    - `001` ~ `030`: Create train datasets
    - `031` ~ `099`: Data augmentation
- `100` ~ `499`: Feature extraction
- `500` ~ `599`: Validation
- `600` ~ `799`: Training
- `800` ~ `899`: Prediction
- `900` ~ `999`: Visualization

# Installation

1. Put input files to `./input`
2. Create feather files (`000`)

```
python src/000_to_feather.py
```

3. Create train datasets (`001`~`030`)

```
python src/001_naive_split.py
```

4. Apply data augmentation to train datasets (`030`~`099`)

```
TODO
```

5. Extract features by specifying the index of the input datasets (`100`~`499`)

```
python src/100_simple_aggregation.py 001
```

`001`: index of the input dataset (`./input/001`)

6. Create validations (`500`~`599`)

```
python src/500_naive_kfolds.py 500_001
```

`500_001`: index of the config json (`./config/500/500_001.json`)

7. Train a model using an input config file (`600`~`799`)

```
TODO
```

8. Predict test files using a input model (`800`~`899`)

```
TODO
```

# Dependency

Please read `./dependency.txt`
