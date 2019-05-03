import gc
import warnings
from log_writer import create_loggers, stop_watch
warnings.filterwarnings('ignore')


@stop_watch
def main():
    pass


if __name__ == "__main__":
    gc.enable()
    create_loggers()
    main()
