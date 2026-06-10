# Mahalanobis Remake1 実験資料

## 1. 目的

`mahalanobis_remake1` は、`each_mask/` 内の各時系列データが実データらしいか、生成データらしいかを判別するための異常検知ベースの実験である。

`train_data/train_sp500_us10y.csv` に含まれる実データを参照分布として用い、各 `each_mask/*.csv` の20次元特徴量ベクトルが、その参照分布からどの程度離れているかを Mahalanobis距離で評価する。

## 2. 現状うまくいっている点

正解が以下であるとする。

```text
Brown: mask2, mask4
SABR : mask2, mask5
```

`mahalanobis_remake1` では、Mahalanobis距離が小さい順に以下の4件が並んだ。

```text
mask2_brown.csv  4.018166
mask4_brown.csv  4.033254
mask2_sabr.csv   4.794737
mask5_sabr.csv   4.966569
```

つまり、実データらしいと判定すべき4件が、距離の小さい4件として正しく抽出されている。

Hotelling's T² による統計的評価でも、同じ4件は `p >= 0.05` となり、実データ参照分布から来たという帰無仮説を棄却できなかった。

```text
mask2_brown.csv  p = 0.857075
mask4_brown.csv  p = 0.852650
mask2_sabr.csv   p = 0.546404
mask5_sabr.csv   p = 0.467452
```

一方で、それ以外の6件はすべて `p < 0.01` となり、実データ参照分布から来たとは考えにくいと評価された。

この結果から、今回の20特徴量は、実データらしい時系列と生成データらしい時系列の差をかなりよく捉えていると考えられる。

## 3. 使用している特徴量

使用している特徴量は20個である。大きく以下の4グループに分けられる。

### 3.1 ボラティリティ水準

```text
sp500_std
dgs10_std
```

各系列の標準偏差を表す。

```text
std(x) = sqrt( sum((x_t - mean(x))^2) / (T - 1) )
```

これは、S&P500リターンおよびDGS10差分の変動の大きさを測る特徴量である。

### 3.2 tail の位置を表す分位点

```text
sp500_q01
sp500_q05
sp500_q95
sp500_q99
dgs10_q01
dgs10_q05
dgs10_q95
dgs10_q99
```

分位点は、データを小さい順に並べたときの指定割合の位置にある値である。

```text
q01 = 下位1%
q05 = 下位5%
q95 = 上位95%
q99 = 上位99%
```

これにより、平均や標準偏差だけでは捉えにくい極端な下落、極端な上昇、金利差分の大きな変動を評価する。

origin版では歪度や尖度を用いていたが、これらは少数の極端値に強く引っ張られる。remake1では分位点を用いることで、tail の水準をより安定的に表現している。

### 3.3 ボラティリティクラスタリング

```text
sp500_abs_autocorr_lag1
sp500_abs_autocorr_lag5
sp500_abs_autocorr_lag20
dgs10_abs_autocorr_lag1
dgs10_abs_autocorr_lag5
dgs10_abs_autocorr_lag20
```

各系列の絶対値系列に対する自己相関である。

```text
abs_autocorr_lagk = corr(|x_t|, |x_{t-k}|)
```

金融時系列では、リターン自体の自己相関は弱くても、絶対値や二乗値には自己相関が残ることが多い。これはボラティリティクラスタリングと呼ばれる。

生成データは、分布の平均や分散を再現できても、ボラティリティが一定期間まとまって続く性質を十分に再現できない場合がある。そのため、この特徴量群は判別に重要である。

### 3.4 複数時系列間の相関構造

```text
cross_corr
rolling_corr_std_60
corr_down_sp500_q05
corr_up_sp500_q95
```

`cross_corr` は、S&P500リターンとDGS10差分の全期間相関である。

```text
cross_corr = corr(sp500, DGS10)
```

`rolling_corr_std_60` は、60日rolling correlation の標準偏差である。

```text
rolling_corr_t = corr(sp500, DGS10) over 60-day window
rolling_corr_std_60 = std(rolling_corr_t)
```

これは、2系列間の相関が時間とともにどの程度変化しているかを表す。

`corr_down_sp500_q05` は、S&P500リターンが下位5%の日だけを取り出した条件付き相関である。

```text
corr_down_sp500_q05 = corr(sp500, DGS10 | sp500 <= q05(sp500))
```

`corr_up_sp500_q95` は、S&P500リターンが上位5%の日だけを取り出した条件付き相関である。

```text
corr_up_sp500_q95 = corr(sp500, DGS10 | sp500 >= q95(sp500))
```

これらは、市場が大きく下落した局面や大きく上昇した局面で、株と金利がどのように連動するかを測る特徴量である。

## 4. なぜ改善したと考えられるか

origin版では、単一時系列の歪度・尖度、lag1の絶対値自己相関、全期間相関、rolling correlation の標準偏差を中心に用いていた。

remake1では、以下の点を改善した。

```text
1. 歪度・尖度を、より安定した分位点特徴量に置き換えた
2. lag1だけでなく lag5, lag20 のボラティリティクラスタリングを追加した
3. 通常時の相関だけでなく、rolling correlation と条件付き相関を追加した
```

特に効いていると考えられるのは、以下の特徴量群である。

```text
sp500_abs_autocorr_lag5
dgs10_abs_autocorr_lag5
dgs10_abs_autocorr_lag20
sp500_abs_autocorr_lag1
rolling_corr_std_60
```

これらは、単なる分布の形ではなく、時系列としての持続性や、2資産間の関係の時間変化を表す。

実データでは、ボラティリティが一定期間続いたり、株と金利の相関が局面によって変化したりする。一方で、生成データはこのような時間方向の構造や複数系列間の揺らぎを再現しきれないことがある。

そのため、remake1では生成データらしいmaskが大きな距離を持ち、実データらしいmaskが小さな距離を持つようになったと考えられる。

## 5. Mahalanobis距離の数式

各 `each_mask` データから抽出した20次元特徴量ベクトルを `x` とする。

実データ参照window群から推定した平均ベクトルを `μ`、標本共分散行列を `Σ` とする。

Mahalanobis距離は以下で定義する。

```text
D_M(x) = sqrt( (x - μ)' Σ^{-1} (x - μ) )
```

通常のユークリッド距離と異なり、Mahalanobis距離は特徴量間の相関とスケールを考慮する。

例えば、分位点同士や標準偏差と分位点は強く相関しやすい。Mahalanobis距離では、こうした共分散構造を踏まえて「実データ参照分布から見て自然な方向のズレか、不自然な方向のズレか」を評価する。

## 6. 実装の流れ

実装は [run_mahalanobis_remake1.py](/Users/satouhinata/Documents/DSS_code/mahalanobis_remake1/scripts/run_mahalanobis_remake1.py) にある。

処理の流れは以下である。

```text
1. train_data/train_sp500_us10y.csv を読み込む
2. 1260日windowを126日strideで切り出す
3. 各windowから20特徴量を抽出する
4. 107個の参照window特徴量から平均ベクトル μ を計算する
5. 107個の参照window特徴量から標本共分散行列 Σ を計算する
6. Gauss-Jordan法で Σ^{-1} を計算する
7. each_mask/*.csv から同じ20特徴量を抽出する
8. 各maskについて Mahalanobis距離を計算する
9. 距離CSV、特徴量CSV、z-score CSV、SVGグラフ、summaryを出力する
```

### 6.1 参照window

参照元である `train_data/train_sp500_us10y.csv` は14,734日分の実データを持つ。

このデータから、以下の条件で参照windowを作成している。

```text
window length = 1260日
stride        = 126日
reference windows = 107個
```

window長は各 `each_mask` データと同じ1260日である。strideは参照windowをずらす幅であり、window長ではない。

### 6.2 特徴量抽出

特徴量抽出は `extract_features()` で行っている。

主な対応関係は以下である。

```text
sample_std()                         -> 標準偏差
quantile(xs, q)                      -> q分位点
autocorr(abs_series, lag)            -> 絶対値系列のlag自己相関
correlation(sp500, dgs10)            -> 2系列間相関
rolling_corr_values(..., window=60)  -> 60日rolling correlation
conditional_corr_by_sp500_quantile() -> S&P500上下5%局面の条件付き相関
```

### 6.3 共分散行列と逆行列

標本共分散行列は `covariance_matrix()` で計算している。

```text
Σ_ij = sum((x_i - μ_i)(x_j - μ_j)) / (n - 1)
```

逆行列は `invert_matrix()` で Gauss-Jordan 法により計算している。

今回は robust covariance や shrinkage covariance は使用していない。通常の標本平均と通常の標本共分散のみを使用している。

### 6.4 距離計算

距離計算は `mahalanobis_distance()` で行っている。

コード上では、まず差分ベクトルを作る。

```text
diff = x - μ
```

次に、逆共分散行列をかける。

```text
transformed = Σ^{-1} diff
```

最後に二次形式を計算して平方根を取る。

```text
squared = diff' transformed
D_M = sqrt(squared)
```

## 7. Hotelling's T² 評価

Hotelling's T² 評価は [run_hotelling_t2.py](/Users/satouhinata/Documents/DSS_code/mahalanobis_remake1/hotelling_t2/scripts/run_hotelling_t2.py) にある。

Hotelling's T² は、Mahalanobis距離の二乗に相当する。

```text
T² = (x - μ)' Σ^{-1} (x - μ)
```

この `T²` をF統計量に変換し、p値を計算している。

今回の設定は以下である。

```text
n = 107 参照window数
p = 20  特徴量次元
dfn = 20
dfd = 87
```

Phase-II の新規観測評価として、以下の変換を用いている。

```text
F = n * (n - p) / [p * (n + 1) * (n - 1)] * T²
```

p値は、このF統計量が自由度 `(p, n-p)` のF分布でどの程度極端かを表す。

```text
p値が大きい -> 実データ参照分布から来たとしても不自然ではない
p値が小さい -> 実データ参照分布から来たとは考えにくい
```

なお、p値は「実データである確率」そのものではない。

## 8. 使用しているアルゴリズム

この実験で使用している主なアルゴリズムは以下である。

```text
1. Sliding window による参照サンプル作成
2. Stylized facts に基づく特徴量抽出
3. 標本平均ベクトルの推定
4. 標本共分散行列の推定
5. Gauss-Jordan 法による逆行列計算
6. Mahalanobis距離による異常度スコアリング
7. Hotelling's T² による統計的検定
8. SVGによる距離・p値の可視化
```

Python実装は、再現性と実行環境の軽さを優先し、外部ライブラリを使わず標準ライブラリのみで書いている。

## 9. 出力ファイル

主な出力は以下である。

```text
mahalanobis_remake1/features/reference_window_features.csv
mahalanobis_remake1/features/each_mask_features.csv
mahalanobis_remake1/results/mahalanobis_distances.csv
mahalanobis_remake1/results/feature_zscores.csv
mahalanobis_remake1/results/summary.txt
mahalanobis_remake1/figures/mahalanobis_distances.svg
mahalanobis_remake1/figures/feature_zscores_heatmap.svg
mahalanobis_remake1/hotelling_t2/results/hotelling_t2_results.csv
mahalanobis_remake1/hotelling_t2/results/summary.txt
mahalanobis_remake1/hotelling_t2/results/explanation_ja.md
mahalanobis_remake1/hotelling_t2/figures/hotelling_t2_pvalues.svg
```

## 10. 今後の改良余地

現状では、通常の標本平均と標本共分散だけで正解maskをうまく分離できている。

次の段階では、性能そのものを大きく変えるというより、結果の安定性を確認することが重要である。

候補は以下である。

```text
1. shrinkage covariance による共分散推定の安定化
2. robust covariance による外れ値の影響低減
3. 特徴量グループ別 Mahalanobis距離
4. 相関の高すぎる特徴量の整理
5. 参照window長やstrideを変えた感度分析
```

特に最初に試すべきなのは shrinkage covariance である。通常共分散に依存しすぎず、少し共分散推定を安定化しても同じ4件が実データらしく残るなら、今回の結果の説得力はさらに高まる。
