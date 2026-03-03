# Fix: MPS Finding Dropout Caused by Reasons Gate

## Problem

When switching from CPU to MPS (Apple Silicon GPU) for embedding computation,
11 findings detected on CPU were lost on MPS across all 4 benchmark repos.
These findings had high composite scores (0.92--0.97) yet were completely
absent from MPS results.

## Root Cause

The detection pipeline has a structural "reasons gate" in `rollup.py` that
decides whether a finding is created. A finding requires at least one reason:

1. A **FUNC**-type candidate with composite >= 0.92, OR
2. An **EXP**-type candidate with composite >= 0.90, OR
3. **2+ WIN**-type candidates (count-based, ignores score)

The finding score reported in output is the **best** composite across all
candidate types -- but this score is not itself a gate. A single WIN candidate
scoring 0.967 is discarded if there is no FUNC candidate above 0.92 and fewer
than 2 WIN candidates total.

MPS floating-point drift (~0.001--0.005 in embedding similarity) is enough to
push a borderline second WIN candidate below the 0.90 candidate-generation
threshold, dropping the window-hit count from 2 to 1 and eliminating the
finding entirely -- even though the best evidence for that finding scores 0.97.

## Fix

Add a 4th reason: if the best composite score of **any** candidate (regardless
of snippet type) meets the func threshold, the finding is created.

```python
# Before
def _reasons(matches, thresholds):
    ...
    if func_hits and best_score(func_hits) >= thresholds.func:
        reasons.append("func_threshold")
    if exp_hits and best_score(exp_hits) >= thresholds.exp:
        reasons.append("exp_threshold")
    if len(win_hits) >= thresholds.min_window_hits:
        reasons.append("min_window_hits")

# After
def _reasons(matches, thresholds):
    ...
    if func_hits and best_score(func_hits) >= thresholds.func:
        reasons.append("func_threshold")
    if exp_hits and best_score(exp_hits) >= thresholds.exp:
        reasons.append("exp_threshold")
    if len(win_hits) >= thresholds.min_window_hits:
        reasons.append("min_window_hits")
    if not reasons and best_score(matches) >= thresholds.func:
        reasons.append("high_score")
```

The `high_score` reason only fires when no other reason already passed,
keeping existing behavior unchanged for findings that were already detected.
It acts as a safety net: if any single candidate is strong enough to meet
the strictest threshold (func), that evidence alone is sufficient.

## Impact

- Recovers all 11 findings lost to MPS float drift
- Also catches real duplicates previously missed by the strict type-specific
  gate (including some with 1.0 similarity scores that were silently dropped)
- Device-independent: results are now stable across CPU/MPS/CUDA backends
