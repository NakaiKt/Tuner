# Tuner
チューニングツール郡

# Install
### 必要なモジュールのインストール
```
$ pip install -r requirements.txt
```

### Utilityのインストール
git cloneしたディレクトリに移動して、以下のコマンドを実行
```
cd Tuner/src
git clone https://github.com/NakaiKt/Utility.git
```

# 準備
get_score_for_tuning関数を作成する
要件は以下

- 引数
    - image: 画像 (cv2.imreadで読み込んだ画像)
    - confidence_threshold: confidence threshold (only detection)
    - iou_threshold: iou threshold (only detection)
- 返り値
    - 予測値 (float or int)

# 使用方法
```python
# 調整可能変数を変更するときは引数として指定
tuner = TuningByOptuna(get_score_for_tuning)# , study_name="study")
study = tuner.main()
```

# 調整可能変数一覧
<!-- 調整可能な値をテーブル形式で表示 -->

|key|type|default|description|
|:--|:--|:--|:--|
|study_name|str|study|チューニングの名前|
|epochs|int|1|学習のエポック数|
|n_trials|int|30|チューニングの試行回数|
|n_warmup_steps|int|5|チューニングの最低試行エポック数|
|direction|str|minimize|チューニングの最適化方向|
|tuning_mode|str|pretrain|チューニングのモード, pretrain or train|
|image_dtype|str|PIL|画像のデータ型, PIL or cv2|
|source|str|./data/images/|画像データのパス|
|csv_path|str|./data/label.csv|ラベルデータのパス|
|model_task|str|detection|モデルのタスク, classification or detection|
|input_width|str|1280, 640|入力画像の幅 [max, min]|
|input_height|str|720, 480|入力画像の高さ [max, min]|
|confidence_threshold|str|0.25, 0.25|検出の信頼度閾値 [max, min]|
|iou_threshold|str|0.45, 0.45|検出のIOU閾値 [max, min]|
|image_type|str|image|get_score_for_tuningが処理する画像のタイプ, image or image_source_list|