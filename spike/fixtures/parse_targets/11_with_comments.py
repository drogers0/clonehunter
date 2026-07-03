def process_data(data):
    # Validate input
    if not data:
        return []

    # Transform each item
    result = []
    for item in data:
        # Skip None values
        if item is None:
            continue
        result.append(item)

    # Return processed data
    return result


def merge_configs(base, override):
    # Start with base
    merged = dict(base)
    # Apply overrides (None values are dropped)
    for key, value in override.items():
        if value is not None:
            merged[key] = value  # overwrite
    return merged
