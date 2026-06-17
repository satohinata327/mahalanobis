# Mahalanobis Remake1

2変量金融時系列データから stylized facts 特徴量を抽出し、Mahalanobis距離と Hotelling's T² によって、各 mask データが実データらしいか生成データらしいかを評価する実験ディレクトリです。

## 1. ディレクトリ構成

```text
mahalanobis_remake1/
  preprocessing/
    data/
      mixed_brown_masked.csv
      mixed_sabr_masked.csv
    scripts/
      create_each_mask.py
  scripts/
    run_mahalanobis_remake1.py
  hotelling_t2/
    scripts/
      run_hotelling_t2.py
  features/
  results/
  figures/
  experiment_documentation.md
```

詳しい実験内容、特徴量、数式、実装の説明は [experiment_documentation.md](/Users/satouhinata/Documents/DSS_code/mahalanobis_remake1/experiment_documentation.md) にまとめています。

## 2. 入力データ

この実験では、以下の入力を使います。

```text
train_data/train_sp500_us10y.csv
preprocessing/data/mixed_brown_masked.csv
preprocessing/data/mixed_sabr_masked.csv
```

`train_data/train_sp500_us10y.csv` は実データ参照分布を作るためのデータです。

`mixed_brown_masked.csv` と `mixed_sabr_masked.csv` は、以下のような列を持つ mixed 形式のCSVです。

```text
mask1_sp500,mask1_DGS10,mask2_sp500,mask2_DGS10,...
```

前処理スクリプトにより、この mixed 形式から `each_mask/` 形式を作成します。

## 3. each_mask の作成

`mahalanobis_remake1` ディレクトリに移動します。

```bash
cd mahalanobis_remake1
```

前処理スクリプトを実行します。

```bash
python3 preprocessing/scripts/create_each_mask.py
```

デフォルトでは、以下を入力として読みます。

```text
preprocessing/data/mixed_brown_masked.csv
preprocessing/data/mixed_sabr_masked.csv
```

そして、以下のようなファイルを `each_mask/` に出力します。

```text
each_mask/mask1_brown.csv
each_mask/mask2_brown.csv
each_mask/mask3_brown.csv
each_mask/mask4_brown.csv
each_mask/mask5_brown.csv
each_mask/mask1_sabr.csv
each_mask/mask2_sabr.csv
each_mask/mask3_sabr.csv
each_mask/mask4_sabr.csv
each_mask/mask5_sabr.csv
```

各CSVは以下の2列を持ちます。

```text
sp500,DGS10
```

入力・出力ディレクトリを変えたい場合は、以下のように指定できます。

```bash
python3 preprocessing/scripts/create_each_mask.py \
  --input-dir preprocessing/data \
  --output-dir each_mask
```

## 4. Mahalanobis距離の実行

`each_mask/` を作成したあと、Mahalanobis距離による評価を実行します。

```bash
python3 scripts/run_mahalanobis_remake1.py \
  --train-csv ../train_data/train_sp500_us10y.csv \
  --mask-dir each_mask \
  --output-dir .
```

主な出力は以下です。

```text
features/reference_window_features.csv
features/each_mask_features.csv
results/reference_window_distances.csv
results/mahalanobis_distances.csv
results/mask_distance_positions.csv
results/feature_zscores.csv
results/summary.txt
figures/mahalanobis_distances.svg
figures/mask_positions_in_reference_distribution.svg
figures/feature_zscores_heatmap.svg
```

`results/mask_distance_positions.csv` では、各 `each_mask` のMahalanobis距離だけでなく、実データ参照windowの距離分布の中でどの位置にあるかも確認できます。

```text
reference_percentile
empirical_upper_tail_probability
```

`reference_percentile` は、参照windowのうち対象mask以下の距離を持つものの割合です。

`empirical_upper_tail_probability` は、参照windowのうち対象mask以上の距離を持つものの割合です。

## 5. Hotelling's T² の実行

Mahalanobis距離の実行後、同じ20特徴量を用いて Hotelling's T² による統計的評価を行えます。

```bash
python3 hotelling_t2/scripts/run_hotelling_t2.py \
  --reference-features features/reference_window_features.csv \
  --mask-features features/each_mask_features.csv \
  --output-dir hotelling_t2
```

主な出力は以下です。

```text
hotelling_t2/results/hotelling_t2_results.csv
hotelling_t2/results/summary.txt
hotelling_t2/results/explanation_ja.md
hotelling_t2/figures/hotelling_t2_pvalues.svg
```

Hotelling's T² では、以下の帰無仮説を検定します。

```text
H0:
対象maskの20次元特徴量ベクトルは、
実データ参照window群と同じ多変量分布に属している。
```

`p < 0.05` の場合、実データ分布から来たという仮説を棄却し、生成データらしいと判断します。

## 6. 使用している20特徴量

S&P500単体の特徴量:

```text
sp500_std
sp500_q01
sp500_q05
sp500_q95
sp500_q99
sp500_abs_autocorr_lag1
sp500_abs_autocorr_lag5
sp500_abs_autocorr_lag20
```

DGS10単体の特徴量:

```text
dgs10_std
dgs10_q01
dgs10_q05
dgs10_q95
dgs10_q99
dgs10_abs_autocorr_lag1
dgs10_abs_autocorr_lag5
dgs10_abs_autocorr_lag20
```

2系列間の特徴量:

```text
cross_corr
rolling_corr_std_60
corr_down_sp500_q05
corr_up_sp500_q95
```

## 7. 現在の結果

現状の結果では、実データ参照windowのMahalanobis距離は以下の範囲にあります。

```text
min  = 3.209714
mean = 4.391009
std  = 0.732900
max  = 7.107789
```

正解realである以下の4件は、この参照window距離分布の中に収まっています。

```text
mask2_brown.csv
mask4_brown.csv
mask2_sabr.csv
mask5_sabr.csv
```

それ以外の6件は参照windowの最大距離を超えており、実データ分布から見て外側に位置しています。

