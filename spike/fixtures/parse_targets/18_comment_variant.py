def process_data(data):
    # First, check if data is empty before processing
    if not data:
        return []

    # Apply the transformation logic
    result = []
    for item in data:
        # Filter out None entries from input
        if item is None:
            continue
        result.append(item)

    # All done, return results
    return result


def merge_configs(base, override):
    # Initialize with base configuration
    merged = dict(base)
    # Walk override entries
    for key, value in override.items():
        if value is not None:
            merged[key] = value  # update in place
    return merged
