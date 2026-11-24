"""Check score summaries, item analysis, reliability, and CLO calculations."""

import numpy as np

from exam_src.analytics.item_analysis import (
    difficulty_label,
    discrimination_label,
    exam_summary_stats,
    kr20,
)


def test_difficulty_index_hand_computed():
    """p = correct / answered. 7 correct out of 10 answered -> p = 0.70."""
    correct, total = 7, 10
    p = correct / total
    assert p == 0.70
    assert difficulty_label(p) == "ค่อนข้างง่าย"  # 0.60-0.80 band


def test_difficulty_label_bands():
    assert difficulty_label(0.85) == "ง่ายมาก"
    assert difficulty_label(0.70) == "ค่อนข้างง่าย"
    assert difficulty_label(0.50) == "เหมาะสม"
    assert difficulty_label(0.30) == "ค่อนข้างยาก"
    assert difficulty_label(0.10) == "ยากมาก"


def test_discrimination_index_hand_computed():
    """Check discrimination using 10 high scorers and 10 low scorers."""
    high_correct, low_correct, n_group = 8, 3, 10
    d = (high_correct - low_correct) / n_group
    assert d == 0.5
    assert discrimination_label(d) == "ดีมาก"


def test_discrimination_negative_flags_bad_key():
    """More of the low group got it right than the high group -- the plan
    calls this a signal to check the answer key itself."""
    d = (2 - 6) / 10
    assert d < 0
    assert "ตรวจสอบเฉลย" in discrimination_label(d)


def test_kr20_hand_computed_perfect_consistency():
    """Equal total scores have zero variance, so KR-20 is undefined."""
    matrix = np.array(
        [
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 1],
        ]
    )
    r = kr20(matrix)
    assert np.isnan(r)  # every student scores 2/4 -> zero variance in total score -> undefined


def test_kr20_hand_computed_varying_scores():
    """4 items, 6 students with varying totals -- cross-check against a
    hand computation of the KR-20 formula."""
    matrix = np.array(
        [
            [1, 1, 1, 1],  # 4
            [1, 1, 1, 0],  # 3
            [1, 1, 0, 0],  # 2
            [1, 0, 0, 0],  # 1
            [0, 0, 0, 0],  # 0
            [1, 1, 1, 1],  # 4
        ]
    )
    k = 4
    p = matrix.mean(axis=0)
    q = 1 - p
    total = matrix.sum(axis=1)
    var_total = total.var(ddof=1)
    expected = (k / (k - 1)) * (1 - (p * q).sum() / var_total)
    assert kr20(matrix) == pytest_approx(expected)


def pytest_approx(x, tol=1e-9):
    import pytest

    return pytest.approx(x, abs=tol)


def test_exam_summary_stats_matches_numpy():
    scores = [60.0, 70.0, 80.0, 90.0, 100.0]
    s = exam_summary_stats(scores)
    assert s.n == 5
    assert s.mean == 80.0
    assert s.median == 80.0
    assert s.minimum == 60.0
    assert s.maximum == 100.0
    assert s.sd == np.std(scores, ddof=1)
