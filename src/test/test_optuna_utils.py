# test for optuna_utils.py
# use pytest

import pytest


from optuna_utils import *

def test_get_filename():
    # get_filenameの正常テスト
    assert get_filename(path="test.csv") == "test"
    assert get_filename(path="test/test.csv") == "test"
    assert get_filename(path="test/test/test.csv") == "test"

def test_get_label_from_csv():
    # get_label_from_csvの正常テスト
    assert get_label_from_csv(image_name="1", csv_path="test.csv") == 5
    assert get_label_from_csv(image_name="2", csv_path="test.csv") == 3
    assert get_label_from_csv(image_name="3", csv_path="test.csv") == 4

def test_get_label_from_csv_failed():
    # get_label_from_csvの異常テスト
    with pytest.raises(IndexError):
        get_label_from_csv(image_name=4, csv_path="test.csv")

def test_get_label_from_file_name():
    # get_label_from_file_nameの正常テスト
    assert get_label_from_file_name(file_path="1.jpg", csv_path="test.csv") == 5
    assert get_label_from_file_name(file_path="test/2.jpg", csv_path="test.csv") == 3
    assert get_label_from_file_name(file_path="test/test/3.png", csv_path="test.csv") == 4