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

def leaky_RMSE(y_true, y_pred):
    """RMSEの計算
    Ground truthの値が大きいほどRMSEを過小評価する

    Args:
        y_true (numpy.ndarray): 正解値
        y_pred (numpy.ndarray): 予測値

    Returns:
        float: RMSE
    """
    # y_trueが0の場合は1に置き換える
    weight = np.where(y_true == 0, 1, y_true)
    return np.sqrt(np.mean((y_true - y_pred) ** 2 / weight))