# test for optuna_utils.py
# use pytest

import pytest
from optuna_args import Args

def test_optuna_args():
    args = Args(study_name = "test", n_trials = 100, test_key = "ttt")
    assert args.study_name == "test"
    assert args.n_trials == 100
    assert args.confidence_threshold == "0.25, 0.25"
    assert args.tuning_mode == "pretrain"

    with pytest.raises(AttributeError):
        assert args.test_key == "ttt"

def test_optuna_args_variable_argument():
    kwargs = {"study_name": "test", "n_trials": 100, "test_key": "ttt"}
    args = Args(**kwargs)
    assert args.study_name == "test"
    assert args.n_trials == 100
    assert args.confidence_threshold == "0.25, 0.25"
    assert args.tuning_mode == "pretrain"

    with pytest.raises(AttributeError):
        assert args.test_key == "ttt"