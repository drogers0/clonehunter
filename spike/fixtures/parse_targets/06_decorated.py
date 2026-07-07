def decorator(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


@decorator
def my_function(x):
    return x * 2


@decorator
def another_function(x, y):
    """Documented decorated function."""
    return x + y
