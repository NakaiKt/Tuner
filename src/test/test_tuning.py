# test for tuning.py
# using pytest
import __init__
import pytest
from tuning import TuningByOptuna


def get_score_for_tuning(image, confidence_threshold=1, iou_threshold=1):
    weight = confidence_threshold * iou_threshold
    # image sizeが1280, 480で最大のスコアを返す
    if image.shape[0] == 1280 and image.shape[1] == 480:
        return 100 * weight
    elif image.shape[0] == 640 and image.shape[1] == 480:
        return 50 * weight
    elif image.shape[0] == 1280 and image.shape[1] == 720:
        return 10 * weight
    else:
        return 1 * weight


def test_main():
    turner = TuningByOptuna(get_score_for_tuning)
    turner.main()

if __name__ == "__main__":
    test_main()
