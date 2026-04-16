# test/test_get_slope.py
from __future__ import annotations

import pytest

from benchmarks.noisy_search import get_nearest_slope


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (4, 0.4),  # exact match
        (3, 0.4),  # below smallest key -> smallest
        (5, 0.8),  # prefer positive distance over negative distance
        (7, 0.8),  # closer to 8 than 4 -> 8
        (8, 0.8),  # prefer positive distance over negative distance
        (6, 0.8),  # tie (4 and 8 equally close) -> prefer higher (8)
        (99, 1.6),  # above largest key -> largest
    ],
)
def test_get_slope_parameterized(n: int, expected: float) -> None:
    slope_map = {4: 0.4, 8: 0.8, 16: 1.6}
    assert get_nearest_slope(n, slope_map) == expected


def test_get_slope_edge_cases() -> None:
    with pytest.raises(IndexError):
        get_nearest_slope(1, {})
    assert get_nearest_slope(0, {1: 0.4}) == 0.4
    assert get_nearest_slope(2, {1: 0.4}) == 0.4
