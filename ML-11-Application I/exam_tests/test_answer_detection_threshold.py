import pytest

from exam_src.vision.answer_detection import decide

LABELS = ["ก", "ข", "ค", "ง"]


def test_unique_maximum_at_30_percent_is_accepted_without_review():
    assert decide([0.30, 0.29, 0.10, 0.05], LABELS) == ("ก", pytest.approx(0.30), "OK")


def test_unique_maximum_above_30_percent_is_accepted_without_review():
    assert decide([0.12, 0.48, 0.31, 0.05], LABELS) == ("ข", pytest.approx(0.48), "OK")


def test_maximum_below_30_percent_still_requires_review():
    answer, confidence, status = decide([0.299, 0.10, 0.05, 0.02], LABELS)
    assert answer is None
    assert confidence == pytest.approx(0.299)
    assert status == "REVIEW"


def test_tied_maximum_at_30_percent_is_not_auto_accepted():
    answer, confidence, status = decide([0.30, 0.30, 0.05, 0.02], LABELS)
    assert answer is None
    assert confidence == pytest.approx(0.30)
    assert status == "MULTIPLE"
