import gc
import warnings
from log_writer import create_loggers, stop_watch
from processor_factory import ProcessorFactory
warnings.filterwarnings('ignore')


@stop_watch
def main():
    ProcessorFactory.process()


if __name__ == "__main__":
    gc.enable()
    create_loggers()
    main()
