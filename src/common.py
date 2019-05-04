import os
import sys
import numpy as np
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


def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != 'object' and col_type != 'datetime64[ns]':
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)  # feather-format cannot accept float16
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage: {:.2f} MB -> {:.2f} MB (Decreased by {:.1f}%)'.format(
        start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
