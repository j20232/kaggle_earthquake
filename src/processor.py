import json
from feature_extractor import FeatureExtractor
from regressor import Regressor
from sampler import Sampler
from log_writer import stop_watch
from pathlib import Path
from utils import get_version


class Processor():
    @classmethod
    @stop_watch
    def process(cls):
        root_path = Path(__file__).absolute().parents[1]
        config_file = list(root_path.glob(
            "config/" + get_version() + "*.json"))[0]
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
