# 二重降下現象の研究で使用しているコードです。公開用なのでここには載せていないコードが多数あります
double_descent
二重降下現象を観察するためのソースコードが様々含まれている

dd.py

二重降下現象をpytorchのライブラリから引っ張ってきたResNetなどを使って観察するコード

dd_scratch_models_combine.py

学習過程の概念獲得を観察するコード
combineのラベルノイズに関して、色・数字のラベル両方異なるラベルにする
accuracyに関して、combineの正解率だけでなく、色・数字の正解率も出力する

dd_scratch_models_for_spread_noise.py

combineのラベルノイズに関して、色・数字のラベル両方異なるラベルにする
accuracyに関して、combineの正解率だけでなく、色・数字の正解率も出力する
ラベルノイズの正解率とラベルノイズでない正解率を出力する
平均・分散も出力する
