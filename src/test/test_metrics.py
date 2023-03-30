# test for metrics.py
# use pytest
from metrics import *
import pytest
import numpy as np


def test_RMSE():
    # test data
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 4, 5])
    # test
    assert RMSE(y_true, y_pred) == 0
