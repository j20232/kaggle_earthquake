import gc
import warnings
from log_writer import create_loggers, stop_watch
from processor import Processor
warnings.filterwarnings('ignore')


@stop_watch
def main():
    Processor.process()


if __name__ == "__main__":
    gc.enable()
    create_loggers()
    main()
