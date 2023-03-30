"""
modelのハイパーパラメータの調整を行うスクリプト

# 準備
### pretrainモデルを用いる場合
get_score_for_tuning関数を作成する必要がある
    get_score_for_tuning要件
        引数
            image: 画像 (cv2.imreadで読み込んだ画像)
            confidence_threshold: confidence threshold (only detection)
            iou_threshold: iou threshold (only detection)
        返り値
            予測値 (float or int)

"""


import argparse
import glob
import logging
import os

import cv2
import numpy as np
import optuna
from metrics import RMSE
from optuna_utils import get_label_from_file_name

from Utility.format import HelpFormatter, setting_logging_config
from Utility.convert import convert_str_to_list
from Utility.validation import validate_in_list

MODEL_TASK = ["detection", "crowd_counting"]
TUNING_MODE = ["pretrain", "train"]
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

# loggingの設定
setting_logging_config()
logger = logging.getLogger(__name__)
optuna.logging.disable_default_handler()

# コマンドライン引数を取得する関数
def get_args_optuna():
    """optunaの設定をコマンドラインから取得する関数

    Returns:
        argparse.Namespace: コマンドライン引数
    """
    parser = argparse.ArgumentParser(formatter_class=HelpFormatter)
    parser.add_argument('--study_name', type=str, default='study', help='study name')
    parser.add_argument('--n_trials', type=int, default=10, help='number of optuna trials')
    parser.add_argument("--n_warmup_steps", type=int, default=5, help="number of warmup steps")
    parser.add_argument("--direction", type=str, default="minimize", help="direction of optimization")
    parser.add_argument("--tuning_mode", type=str, default="pretrain", help=f"tuning mode. {TUNING_MODE}")

    args = parser.parse_args()

    logger.info("study_name: %s", args.study_name)
    logger.info("n_trials: %s", args.n_trials)
    logger.info("n_warmup_steps: %s", args.n_warmup_steps)
    logger.info("direction: %s", args.direction)
    logger.info("tuning_mode: %s", args.tuning_mode)

    return args

def get_args_pretrain():
    """pretrainの調整の設定をコマンドラインから取得する関数

    Returns:
        argparse.Namespace: コマンドライン引数
    """
    parser = argparse.ArgumentParser(formatter_class=HelpFormatter)
    parser.add_argument("--epoch", type=int, default=15, help="number of epoch")
    parser.add_argument("--source", type=str, default="./data/images", help="source of images")
    parser.add_argument("--csv_path", type=str, default="./data/label.csv", help="path of label csv file")
    parser.add_argument("--model_task", type=str, default="detection", help=f"task of model. {MODEL_TASK}")
    parser.add_argument("--input_width", type=str, default="1280, 640", help="input width. When tuning, specify as '1280,640'")
    parser.add_argument("--input_height", type=str, default="720, 480", help="input height, When tuning, specify as '720,480'")
    parser.add_argument("--confidence_threshold", type=str, default="0.25, 0.25", help="confidence threshold. When tuning, [min, max]")
    parser.add_argument("--iou_threshold", type=str, default="0.45, 0.45", help="iou threshold. When tuning, [min, max]")

    args = parser.parse_args()

    logger.info("epoch: %s", args.epoch)
    logger.info("source: %s", args.source)
    logger.info("csv_path: %s", args.csv_path)
    logger.info("model_task: %s", args.model_task)
    logger.info("input_width: %s", args.input_width)
    logger.info("input_height: %s", args.input_height)
    logger.info("confidence_threshold: %s", args.confidence_threshold)
    logger.info("iou_threshold: %s", args.iou_threshold)

    return args


class TuningByOptuna:
    """optunaを用いたハイパーパラメータの調整を行うクラス
    """
    def __init__(self, get_score_for_tuning):
        self.get_score_for_tuning = get_score_for_tuning
        self.optuna_args = get_args_optuna()

        if self.optuna_args.tuning_mode == "pretrain":
            self.set_pretrain_parameter()

    def set_pretrain_parameter(self):
        """pretrainの調整の設定を取得する関数
        """
        self.tuning_args = get_args_pretrain()
        # モデル設定
        self.epoch = self.tuning_args.epoch
        self.image_source = self.tuning_args.source
        self.model_task = self.tuning_args.model_task

        if not validate_in_list(self.model_task, MODEL_TASK):
            # モデルタスクが不正な場合はエラー
            raise ValueError(f"model_task must be {MODEL_TASK}")

    def get_scores_for_pretrain_tuning(self,
                                       image_source: str,
                                       input_width: int,
                                       input_height: int,
                                       confidence_threshold: float,
                                       iou_threshold: float,
                                       csv_path: str,
                                       model_task: str) -> float:
        """pretrainの調整のための評価値の計算

        Args:
            image_source (str): 画像のパス
            input_width (int): 入力画像の幅
            input_height (int): 入力画像の高さ
            confidence_threshold (float): confidence threshold
            iou_threshold (float): iou threshold
            csv_path (str): labelの入ったcsvのパス
            model_task (str): モデルのタスク

        Returns:
            float: 評価値
        """
        y_pred_list = []
        y_label_list = []
        image_source = os.path.join(image_source + "/*")
        for image_name in glob.glob(image_source):
            image_name = image_name.replace("\\", "/")
            # 画像の読み込み
            image = cv2.imread(image_name)
            # 画像サイズ(幅, 高さ)
            image = cv2.resize(image, (input_width, input_height))
            # 予測
            if model_task == "detection":
                y_pred = self.get_score_for_tuning(image = image, confidence_threshold = confidence_threshold, iou_threshold = iou_threshold)
            elif model_task == "crowd_counting":
                y_pred = self.get_score_for_tuning(image = image)
            y_pred_list.append(y_pred)

            # csvからimage_nameのラベル取得
            y_label_list.append(get_label_from_file_name(file_path = image_name, csv_path=csv_path))

        # 評価値の計算
        # listをnumpyに変換
        y_pred_list = np.array(y_pred_list)
        y_label_list = np.array(y_label_list)
        score = RMSE(y_label_list, y_pred_list)

        return score

    def objective_pretrain(self, trial):
        """pretrainの調整の目的関数

        Args:
            trial (optuna.trial.Trial): optunaのtrial

        Returns:
            float: 評価値
        """

        # ハイパーパラメータ設定
        input_width = trial.suggest_categorical("input_width", convert_str_to_list(self.tuning_args.input_width, format="int"))
        input_height = trial.suggest_categorical("input_height", convert_str_to_list(self.tuning_args.input_height, format="int"))

        if self.model_task == MODEL_TASK[0]:
            # detectionの場合
            # 文字列をリストに変換
            confidence_threshold_list = convert_str_to_list(self.tuning_args.confidence_threshold, format="float")
            iou_threshold_list = convert_str_to_list(self.tuning_args.iou_threshold, format="float")

            confidence_threshold = trial.suggest_float("confidence_threshold", confidence_threshold_list[0], confidence_threshold_list[1])
            iou_threshold = trial.suggest_float("iou_threshold", iou_threshold_list[0], iou_threshold_list[1])

        for e in range(self.epoch):
            # モデルの調整
            score = self.get_scores_for_pretrain_tuning(image_source = self.image_source, input_width = input_width, input_height = input_height, confidence_threshold = confidence_threshold, iou_threshold = iou_threshold, csv_path = self.tuning_args.csv_path, model_task=self.model_task)
            trial.report(score, e)

            logger.info("epoch: %d, score: %f", e, score)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return score

    def main(self):
        """main関数
        """
        logger.info("start tuning")
        # studyの作成
        study = optuna.create_study(
            pruner = optuna.pruners.MedianPruner(n_warmup_steps = self.optuna_args.n_warmup_steps),
            study_name=self.optuna_args.study_name,
            direction=self.optuna_args.direction)

        if self.optuna_args.tuning_mode == TUNING_MODE[0]:
            # pretrainの場合
            study.optimize(self.objective_pretrain, n_trials=self.optuna_args.n_trials)

        logger.info("best_params: %s", study.best_params)
        logger.info("complete tuning")

        return study

if __name__ == "__main__":
    tuning = TuningByOptuna(get_score_for_tuning=None)
    tuning.main()