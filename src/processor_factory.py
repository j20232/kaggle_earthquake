from log_writer import stop_watch
from pathlib import Path


class ProcessorFactory():

    ROOT_PATH = Path(__file__).absolute().parents[1]

    @classmethod
    @stop_watch
    def process(cls):

        # Split the config file ({version}.json) into each variable
        pass
