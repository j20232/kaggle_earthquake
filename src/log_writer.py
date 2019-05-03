import time
from functools import wraps
from logging import getLogger, Formatter, FileHandler, StreamHandler
from logging import INFO, DEBUG
from pathlib import Path
from utils import get_version


def get_training_logger(version):
    return getLogger(version + "train")


def stop_watch(func):
    @wraps(func)
    def wrapper(*args, **kargs):
        version = get_version()
        start = time.time()
        log = "[S] {}".format(func.__name__)
        getLogger(version).info(log)

        result = func(*args, **kargs)
        elapsed_time = int(time.time() - start)
        minits, sec = divmod(elapsed_time, 60)
        hour, minits = divmod(minits, 60)

        log = "[F] {} :{:0>2}:{:0>2}:{:0>2}".format(func.__name__, hour, minits, sec)
        getLogger(version).info(log)
        return result
    return wrapper


def create_loggers():
    train_formatter = Formatter("[%(levelname)s] %(asctime)s >>\t%(message)s")
    __create_logger(train_formatter, ".log", "main")
    __create_logger(Formatter(), ".tsv", "train")


def __create_logger(formatter, extension, post_fix):
    log_path = Path(__file__).parents[1] / "log" / post_fix
    Path.mkdir(log_path, exist_ok=True, parents=True)
    log_file = Path(log_path / (get_version() + extension)).resolve()

    if post_fix == "train":
        logger_ = get_training_logger(get_version())
    else:
        logger_ = getLogger(get_version())
    logger_.setLevel(DEBUG)

    file_handler = FileHandler(log_file)
    file_handler.setLevel(DEBUG)
    file_handler.setFormatter(formatter)
    logger_.addHandler(file_handler)

    stream_handler = StreamHandler()
    stream_handler.setLevel(INFO)
    stream_handler.setFormatter(formatter)
    logger_.addHandler(stream_handler)
