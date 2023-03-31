import logging

logger = logging.getLogger("optuna")

class Args():
    def __init__(self, **kwargs):
        # インスタンス変数の初期値を設定する
        self.set_default_instance_value()

        for k, v in kwargs.items():
            # インスタンス変数が指定されている場合は上書きする
            if hasattr(self, k):
                setattr(self, k, v)

            # 定義されていないインスタンス変数が指定されている場合はWarning
            else:
                logger.warning("AttributeError: %s is not defined", k)

        self.logging_args()

    def set_default_instance_value(self):
        """インスタンス変数をデフォルトで設定
        """
        self.study_name = "study"
        self.epoch = 1
        self.n_trials = 30
        self.n_warmup_steps = 5
        self.direction = "minimize"
        self.metrics = "RMSE"
        self.tuning_mode = "pretrain"
        self.image_dtype = "PIL"

        self.source = "./data/images/"
        self.csv_path = "./data/label.csv"
        self.model_task = "detection"
        self.input_width = "1280, 640"
        self.input_height = "720, 480"
        self.confidence_threshold = "0.25, 0.25"
        self.iou_threshold = "0.45, 0.45"

    def logging_args(self):
        """インスタンス変数の値をログに出力する関数
        """
        for k, v in self.__dict__.items():
            logger.info("%s: %s", k, v)

    def print_args(self):
        """インスタンス変数の値を出力する関数
        README.mdに出力するための関数
        key (type): value
        """

        print("|key|type|value|")
        for k, v in self.__dict__.items():
            print(f"|{k}|{type(v).__name__}|{v}")

if __name__ == "__main__":
    args = Args()
    args.print_args()