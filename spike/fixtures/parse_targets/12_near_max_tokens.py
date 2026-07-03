def compute_weighted_average(
    values: list[float],
    weights: list[float],
    normalize: bool = True,
    clamp_min: float = 0.0,
    clamp_max: float = 1.0,
) -> float:
    """Compute the weighted average of a list of values with optional normalization.

    Args:
        values: The input values to average.
        weights: Weights corresponding to each value.
        normalize: Whether to normalize weights to sum to 1.
        clamp_min: Minimum clamp value for the result.
        clamp_max: Maximum clamp value for the result.

    Returns:
        The weighted average clamped to [clamp_min, clamp_max].
    """
    if not values or not weights:
        return 0.0
    if len(values) != len(weights):
        raise ValueError(f"Length mismatch: {len(values)} vs {len(weights)}")
    total_weight = sum(weights)
    if total_weight == 0.0:
        return 0.0
    if normalize:
        weights = [w / total_weight for w in weights]
    weighted_sum = sum(v * w for v, w in zip(values, weights))
    result = max(clamp_min, min(clamp_max, weighted_sum))
    return result


def rolling_window_stats(
    series: list[float],
    window: int = 10,
    step: int = 1,
) -> list[dict[str, float]]:
    """Compute rolling window statistics over a time series.

    Returns a list of dicts with keys: mean, std, min, max.
    """
    if window <= 0:
        raise ValueError("window must be positive")
    results = []
    for i in range(0, len(series) - window + 1, step):
        chunk = series[i : i + window]
        n = len(chunk)
        mean = sum(chunk) / n
        variance = sum((x - mean) ** 2 for x in chunk) / n
        std = variance**0.5
        results.append(
            {
                "mean": mean,
                "std": std,
                "min": min(chunk),
                "max": max(chunk),
            }
        )
    return results
