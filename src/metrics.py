import numpy as np

def RMSE(y_true, y_pred):
    """RMSEの計算

    Args:
        y_true (numpy.ndarray): 正解値
        y_pred (numpy.ndarray): 予測値

    Returns:
        float: RMSE
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))