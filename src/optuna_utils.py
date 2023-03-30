"""
Optunaのユーティリティ
"""

from Utility.csv_handler import get_column

def get_filename(path: str) -> str:
    """pathからfilenameを取得

    Args:
        path (str): パス

    Returns:
        str: filename
    """
    filename = path.split("/")[-1].split(".")[0]

    return filename

def get_label_from_csv(image_name: str, csv_path: str):
    """csvからbasenameのラベルを取得
    csvの構造は以下の通り
    image_name, label

    Args:
        image_name (str): 画像名
        csv_path (str): csvのパス

    Returns:
        int: ラベル
    """
    # csvからbasenameのラベル取得
    label = get_column(key_column="image_name", key=image_name, csv_file_path=csv_path)[0]

    return int(label)

def get_label_from_file_name(file_path: str, csv_path: str) -> int:
    """ファイル名からラベルを取得

    Args:
        file_path (str): ファイルパス
        csv_path (str): csvのパス

    Returns:
        int: ラベル
    """
    # 拡張子を除いた画像名の取得
    basename = get_filename(file_path)

    # csvからbasenameのラベル取得
    label = get_label_from_csv(image_name=basename, csv_path=csv_path)

    return label