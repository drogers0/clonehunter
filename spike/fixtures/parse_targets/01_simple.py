def add(x, y):
    result = x + y
    return result


def multiply(a, b):
    return a * b


def divide(numerator, denominator):
    if denominator == 0:
        raise ValueError("cannot divide by zero")
    return numerator / denominator
