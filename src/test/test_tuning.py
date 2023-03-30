# test for tuning.py
# using pytest
import pytest
from tuning import TuningByOptuna


def get_score_for_tuning(image, confidence_threshold=1, iou_threshold=1):
    weight = confidence_threshold * iou_threshold
    # image sizeが480, 1280, 3で最良のスコアを返す
    if image.shape[0] == 480 and image.shape[1] == 1280:
        return 1
    elif image.shape[0] == 480 and image.shape[1] == 640:
        return 90 * weight
    elif image.shape[0] == 720 and image.shape[1] == 1280:
        return 100 * weight
    else:
        return 150 * weight


def test_main():
    turner = TuningByOptuna(get_score_for_tuning)
    study = turner.main()
    assert study.best_params["confidence_threshold"] == 0.25
    assert study.best_params["iou_threshold"] == 0.45
    assert study.best_params["input_width"] == 1280
    assert study.best_params["input_height"] == 480