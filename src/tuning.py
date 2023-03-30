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


import glob
import logging
import os

import cv2
import numpy as np
import optuna
from metrics import RMSE
from optuna_utils import get_label_from_file_name
from optuna_args import Args

from Utility.format import setting_logging_config
from Utility.convert import convert_str_to_list
from Utility.validation import validate_in_list

MODEL_TASK = ["detection", "crowd_counting"]
TUNING_MODE = ["pretrain", "train"]
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

# loggingの設定
setting_logging_config()
logger = logging.getLogger("optuna")
optuna.logging.disable_default_handler()


class TuningByOptuna:
    """optunaを用いたハイパーパラメータの調整を行うクラス
    """
    def __init__(self, get_score_for_tuning, **kwargs):
        self.get_score_for_tuning = get_score_for_tuning
        self.optuna_args = Args(**kwargs)

        if self.optuna_args.tuning_mode == "pretrain":
            self.set_pretrain_parameter()

    def set_pretrain_parameter(self):
        """pretrainの調整の設定を取得する関数
        """
        # モデル設定
        self.epoch = self.optuna_args.epoch
        self.image_source = self.optuna_args.source
        self.model_task = self.optuna_args.model_task

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
        input_width = trial.suggest_categorical("input_width", convert_str_to_list(self.optuna_args.input_width, format="int"))
        input_height = trial.suggest_categorical("input_height", convert_str_to_list(self.optuna_args.input_height, format="int"))

        if self.model_task == MODEL_TASK[0]:
            # detectionの場合
            # 文字列をリストに変換
            confidence_threshold_list = convert_str_to_list(self.optuna_args.confidence_threshold, format="float")
            iou_threshold_list = convert_str_to_list(self.optuna_args.iou_threshold, format="float")

            confidence_threshold = trial.suggest_float("confidence_threshold", confidence_threshold_list[0], confidence_threshold_list[1])
            iou_threshold = trial.suggest_float("iou_threshold", iou_threshold_list[0], iou_threshold_list[1])

        else:
            confidence_threshold = 0.0
            iou_threshold = 0.0

        for e in range(self.epoch):
            # モデルの調整
            score = self.get_scores_for_pretrain_tuning(image_source = self.image_source, input_width = input_width, input_height = input_height, confidence_threshold = confidence_threshold, iou_threshold = iou_threshold, csv_path = self.optuna_args.csv_path, model_task=self.model_task)
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