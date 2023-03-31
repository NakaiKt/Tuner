"""
modelのハイパーパラメータの調整を行うスクリプト

# 準備
### pretrainモデルを用いる場合
get_score_for_tuning関数を作成する必要がある
    get_score_for_tuning要件
        引数
            image: 画像 torch.Tensor
            confidence_threshold: confidence threshold (only detection)
            iou_threshold: iou threshold (only detection)
        返り値
            予測値 (float or int)

"""


import glob
import logging
import os

import numpy as np
import optuna
from metrics import RMSE, leaky_RMSE
from optuna_utils import get_label_from_file_name
from optuna_args import Args

from Utility.format import setting_logging_config
from Utility.convert import convert_str_to_list
from Utility.validation import validate_in_list
from Utility.image_handler import load_image_cv2, resize_image_cv2, load_image_PIL, resize_image_PIL,normalize_image_cv2, normalize_image_PIL

MODEL_TASK = ["detection", "crowd_counting"]
TUNING_MODE = ["pretrain", "train"]
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

# loggingの設定
setting_logging_config(log_name="optuna")
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

    def get_scores_for_pretrain_tuning(self, input_width, input_height, confidence_threshold, iou_threshold ) -> float:
        """pretrainの調整のための評価値の計算

        Args:
            input_width (int): 画像の幅
            input_height (int): 画像の高さ
            confidence_threshold (float): confidence threshold (only detection)
            iou_threshold (float): iou threshold (only detection)

        Returns:
            float: 評価値
        """
        y_pred_list = []
        y_label_list = []
        image_source = os.path.join(self.optuna_args.source + "*")
        for image_name in glob.glob(image_source):
            image_name = image_name.replace("\\", "/")
            # 画像の読み込み
            image = load_image_PIL(image_name) if self.optuna_args.image_dtype == "PIL" else load_image_cv2(image_name)
            # 画像サイズ(幅, 高さ)
            image = resize_image_PIL(image, (input_width, input_height)) if self.optuna_args.image_dtype == "PIL" else resize_image_cv2(image, (input_width, input_height))
            image = normalize_image_PIL(image) if self.optuna_args.image_dtype == "PIL" else normalize_image_cv2(image)
            # 予測
            if self.optuna_args.model_task == "detection":
                y_pred = self.get_score_for_tuning(image = image, confidence_threshold = confidence_threshold, iou_threshold = iou_threshold)
            elif self.optuna_args.model_task == "crowd_counting":
                y_pred = self.get_score_for_tuning(image = image)
            y_pred_list.append(y_pred)

            # csvからimage_nameのラベル取得
            y_label_list.append(get_label_from_file_name(file_path = image_name, csv_path=self.optuna_args.csv_path))

        # 評価値の計算
        # listをnumpyに変換
        y_pred_list = np.array(y_pred_list)
        y_label_list = np.array(y_label_list)
        if self.optuna_args.metrics == "RMSE":
            score = RMSE(y_label_list, y_pred_list)
        elif self.optuna_args.metrics == "LRMSE":
            score = leaky_RMSE(y_label_list, y_pred_list)

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
            score = self.get_scores_for_pretrain_tuning(input_width = input_width, input_height = input_height, confidence_threshold = confidence_threshold, iou_threshold = iou_threshold)
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