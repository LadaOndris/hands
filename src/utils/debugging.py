from functools import wraps
from time import time


def timing(func):
    @wraps(func)
    def wrap(*args, **kw):
        ts = time()
        result = func(*args, **kw)
        te = time()
        elapsed_secs = te - ts
        elapsed_ms = elapsed_secs * 1000
        print('func [{}] took: {:.2f} ms'.format(
            func.__name__, elapsed_ms))
        return result

    return wrap
