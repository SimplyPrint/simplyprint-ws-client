import functools


def exception_as_value(*args, return_default=False, default=None, **kwargs):
    """Internal decorator to return an exception as a value

    Only used to minimize runtime overhead.
    """

    def decorator(func):
        if not callable(func):
            raise ValueError("exception_as_value decorator must be used on a callable")

        @functools.wraps(func)
        def wrapper(*fargs, **fkwargs):
            try:
                return func(*fargs, **fkwargs)
            except Exception as e:
                return e if not return_default else default

        return wrapper

    if args and callable(args[0]):
        return decorator(args[0])

    return decorator
