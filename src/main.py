import gc
import json
import warnings
from feature_extractor import FeatureExtractor
from log_writer import create_loggers, stop_watch
from pathlib import Path
from regressor import Regressor
from sampler import Sampler
from utils import get_version
warnings.filterwarnings('ignore')


@stop_watch
def main():
    root_path = Path(__file__).absolute().parents[1]
    config_file = list(root_path.glob("config/" + get_version() + "*.json"))[0]
    with config_file.open() as f:
        params = json.load(f)

    data_params = root_path / params["Data"]
    sampler_params = params["Sampler"]
    validation_params = params["Validation"]
    sampler = Sampler(data_params, sampler_params, validation_params)
    dataset_path = sampler.save_dataset()

    feature_params = params["Feature"]
    feature_extractor = FeatureExtractor(dataset_path, feature_params)
    feature_path = feature_extractor.extract()

    model_params = params["Model"]
    regressor = Regressor(feature_path, model_params)
    model_path = regressor.train()
    submit_path = regressor.predict(model_path)
    print(submit_path)


if __name__ == "__main__":
    gc.enable()
    create_loggers()
    main()
