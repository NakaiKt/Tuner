# test for tuning.py
# using pytest
import pytest
import numpy as np

from tuning import TuningByOptuna

def get_score_for_tuning(image, confidence_threshold=1, iou_threshold=1):
    """test用のスコアを返す関数

    Args:
        image (numpy.ndarray): 入力画像
        confidence_threshold (int, optional): 検出閾値. Defaults to 1.
        iou_threshold (int, optional): iou閾値. Defaults to 1.

    Returns:
        float: 関数出力値
    """
    weight = confidence_threshold * iou_threshold
    # image sizeが480, 1280, 3で最良のスコアを返す
    if image.size()[2] == 480 and image.size()[3] == 1280:
        return 1
    elif image.size()[2] == 720 and image.size()[3] == 1280:
        return 90 * weight
    elif image.size()[2] == 480 and image.size()[3] == 640:
        return 100 * weight
    else:
        return 150 * weight
    
def get_score_for_tuning_image_source_list(image_source_list, confidence_threshold, iou_threshold, input_width, input_height):
    return np.array([1, 1, 1, 1])


def test_main():
    tuner = TuningByOptuna(get_score_for_tuning)
    study = tuner.main()
    assert study.best_params["confidence_threshold"] == 0.25
    assert study.best_params["iou_threshold"] == 0.45
    # 最適化問題なので，失敗することもある
    assert study.best_params["input_width"] == 1280
    assert study.best_params["input_height"] == 480

def test_get_score_for_pretrain_tuning_image_source_list():
    tuner = TuningByOptuna(get_score_for_tuning_image_source_list)
    score = tuner.get_score_for_pretrain_tuning_image_source_list(input_width=1280, input_height=480, confidence_threshold=0.25, iou_threshold=0.45)
    assert score == 0