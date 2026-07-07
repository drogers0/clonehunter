def apply(func=lambda x: x, value=42):
    return func(value)


def make_adder(n, transform=lambda x: x):
    def adder(x):
        return transform(x) + n

    return adder


def pipeline(data, steps=None):
    if steps is None:
        steps = [lambda x: x]
    result = data
    for step in steps:
        result = step(result)
    return result
