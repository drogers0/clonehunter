def build_index(items):
    x = "outer"
    index = {x: i for i, x in enumerate(items)}
    return index


def nested_comp(matrix):
    return [cell for row in matrix for cell in row if cell > 0]


def flatten_dict(d, prefix=""):
    result = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            result.update(flatten_dict(v, key))
        else:
            result[key] = v
    return result
