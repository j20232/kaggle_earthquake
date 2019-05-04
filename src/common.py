import os
import sys
from datetime import datetime
from functools import wraps
from time import time


def stop_watch(func):
    @wraps(func)
    def wrapper(*args, **kargs):
        start = time()
        log = "[START]  {}: {}() | PID: {} ({})".format(sys.argv[0], func.__qualname__, os.getpid(), datetime.today())
        print(log)
        result = func(*args, **kargs)
        elapsed_time = int(time() - start)
        minits, sec = divmod(elapsed_time, 60)
        hour, minits = divmod(minits, 60)
        log = "[FINISH] {}: {}() | PID: {} ({}) >> [Time]: {:0>2}:{:0>2}:{:0>2}".format(
            sys.argv[0], func.__qualname__, os.getpid(), datetime.today(), hour, minits, sec)
        print(log)
        return result
    return wrapper
