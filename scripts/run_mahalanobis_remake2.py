#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


FEATURE_NAMES = [
    "sp500_mean",
    "sp500_std",
    "sp500_skew",
    "sp500_kurtosis",
    "dgs10_mean",
    "dgs10_std",
    "dgs10_skew",
    "dgs10_kurtosis",
    "sp500_abs_q90",
    "sp500_abs_q95",
    "sp500_abs_q99",
    "sp500_abs_q99_over_std",
    "dgs10_abs_q90",
    "dgs10_abs_q95",
    "dgs10_abs_q99",
    "dgs10_abs_q99_over_std",
    "sp500_autocorr_lag1",
    "sp500_autocorr_lag5",
    "sp500_abs_autocorr_lag1",
    "sp500_abs_autocorr_lag5",
    "dgs10_autocorr_lag1",
    "dgs10_autocorr_lag5",
    "dgs10_abs_autocorr_lag1",
    "dgs10_abs_autocorr_lag5",
    "cross_corr",
    "joint_abs_q95_move_ratio",
]


def read_series(path: Path) -> tuple[list[float], list[float]]:
    sp500: list[float] = []
    dgs10: list[float] = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "sp500" not in reader.fieldnames or "DGS10" not in reader.fieldnames:
            raise ValueError(f"{path} must contain sp500 and DGS10 columns")
        for row in reader:
            try:
                sp = float(row["sp500"])
                dg = float(row["DGS10"])
            except (TypeError, ValueError):
                continue
            if math.isfinite(sp) and math.isfinite(dg):
                sp500.append(sp)
                dgs10.append(dg)
    if len(sp500) < 100:
        raise ValueError(f"{path} has too few usable rows: {len(sp500)}")
    return sp500, dgs10


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def sample_std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mu = mean(xs)
    return math.sqrt(max(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1), 0.0))


def quantile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    pos = q * (len(ys) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ys[lo]
    weight = pos - lo
    return ys[lo] * (1.0 - weight) + ys[hi] * weight


def skewness(xs: list[float]) -> float:
    if len(xs) < 3:
        return 0.0
    mu = mean(xs)
    m2 = sum((x - mu) ** 2 for x in xs) / len(xs)
    if m2 <= 0:
        return 0.0
    m3 = sum((x - mu) ** 3 for x in xs) / len(xs)
    return m3 / (m2 ** 1.5)


def kurtosis(xs: list[float]) -> float:
    if len(xs) < 4:
        return 0.0
    mu = mean(xs)
    m2 = sum((x - mu) ** 2 for x in xs) / len(xs)
    if m2 <= 0:
        return 0.0
    m4 = sum((x - mu) ** 4 for x in xs) / len(xs)
    return m4 / (m2 * m2)


def correlation(xs: list[float], ys: list[float]) -> float:
    n = min(len(xs), len(ys))
    if n < 2:
        return 0.0
    x = xs[:n]
    y = ys[:n]
    mx = mean(x)
    my = mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den_x = sum((a - mx) ** 2 for a in x)
    den_y = sum((b - my) ** 2 for b in y)
    den = math.sqrt(den_x * den_y)
    return num / den if den > 0 else 0.0


def autocorr(xs: list[float], lag: int) -> float:
    if len(xs) <= lag:
        return 0.0
    return correlation(xs[:-lag], xs[lag:])


def joint_abs_q95_move_ratio(sp500: list[float], dgs10: list[float]) -> float:
    abs_sp500 = [abs(x) for x in sp500]
    abs_dgs10 = [abs(x) for x in dgs10]
    sp_threshold = quantile(abs_sp500, 0.95)
    dgs_threshold = quantile(abs_dgs10, 0.95)
    n = min(len(abs_sp500), len(abs_dgs10))
    if n == 0:
        return 0.0
    count = sum(
        1
        for sp, dg in zip(abs_sp500[:n], abs_dgs10[:n])
        if sp >= sp_threshold and dg >= dgs_threshold
    )
    return count / n


def extract_features(sp500: list[float], dgs10: list[float]) -> dict[str, float]:
    abs_sp500 = [abs(x) for x in sp500]
    abs_dgs10 = [abs(x) for x in dgs10]
    sp500_std = sample_std(sp500)
    dgs10_std = sample_std(dgs10)
    features = {
        "sp500_mean": mean(sp500),
        "sp500_std": sp500_std,
        "sp500_skew": skewness(sp500),
        "sp500_kurtosis": kurtosis(sp500),
        "dgs10_mean": mean(dgs10),
        "dgs10_std": dgs10_std,
        "dgs10_skew": skewness(dgs10),
        "dgs10_kurtosis": kurtosis(dgs10),
        "sp500_abs_q90": quantile(abs_sp500, 0.90),
        "sp500_abs_q95": quantile(abs_sp500, 0.95),
        "sp500_abs_q99": quantile(abs_sp500, 0.99),
        "sp500_abs_q99_over_std": quantile(abs_sp500, 0.99) / sp500_std if sp500_std > 0 else 0.0,
        "dgs10_abs_q90": quantile(abs_dgs10, 0.90),
        "dgs10_abs_q95": quantile(abs_dgs10, 0.95),
        "dgs10_abs_q99": quantile(abs_dgs10, 0.99),
        "dgs10_abs_q99_over_std": quantile(abs_dgs10, 0.99) / dgs10_std if dgs10_std > 0 else 0.0,
        "sp500_autocorr_lag1": autocorr(sp500, 1),
        "sp500_autocorr_lag5": autocorr(sp500, 5),
        "sp500_abs_autocorr_lag1": autocorr(abs_sp500, 1),
        "sp500_abs_autocorr_lag5": autocorr(abs_sp500, 5),
        "dgs10_autocorr_lag1": autocorr(dgs10, 1),
        "dgs10_autocorr_lag5": autocorr(dgs10, 5),
        "dgs10_abs_autocorr_lag1": autocorr(abs_dgs10, 1),
        "dgs10_abs_autocorr_lag5": autocorr(abs_dgs10, 5),
        "cross_corr": correlation(sp500, dgs10),
        "joint_abs_q95_move_ratio": joint_abs_q95_move_ratio(sp500, dgs10),
    }
    return {name: features[name] for name in FEATURE_NAMES}


def make_reference_features(
    sp500: list[float], dgs10: list[float], window_length: int, stride: int
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for start in range(0, len(sp500) - window_length + 1, stride):
        end = start + window_length
        row: dict[str, float | int] = extract_features(sp500[start:end], dgs10[start:end])
        row["window_start"] = start
        row["window_end"] = end - 1
        rows.append(row)
    if len(rows) <= len(FEATURE_NAMES):
        raise ValueError("not enough reference windows to estimate covariance")
    return rows


def column_means(rows: list[dict[str, float | int]], names: list[str]) -> list[float]:
    return [mean([float(row[name]) for row in rows]) for name in names]


def covariance_matrix(rows: list[dict[str, float | int]], names: list[str]) -> list[list[float]]:
    n = len(rows)
    mus = column_means(rows, names)
    matrix: list[list[float]] = []
    for i, name_i in enumerate(names):
        values: list[float] = []
        for j, name_j in enumerate(names):
            value = sum((float(row[name_i]) - mus[i]) * (float(row[name_j]) - mus[j]) for row in rows)
            values.append(value / (n - 1))
        matrix.append(values)
    return matrix


def invert_matrix(matrix: list[list[float]]) -> list[list[float]]:
    n = len(matrix)
    aug = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            raise ValueError("covariance matrix is singular")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        pivot_value = aug[col][col]
        aug[col] = [value / pivot_value for value in aug[col]]
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            aug[row] = [value - factor * base for value, base in zip(aug[row], aug[col])]
    return [row[n:] for row in aug]


def mat_vec_mul(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [sum(a * b for a, b in zip(row, vector)) for row in matrix]


def mahalanobis_distance(values: list[float], mus: list[float], inv_cov: list[list[float]]) -> float:
    diff = [x - mu for x, mu in zip(values, mus)]
    transformed = mat_vec_mul(inv_cov, diff)
    return math.sqrt(max(sum(a * b for a, b in zip(diff, transformed)), 0.0))


def empirical_percentile(value: float, reference_values: list[float]) -> float:
    return 100.0 * sum(1 for ref in reference_values if ref <= value) / len(reference_values)


def empirical_upper_tail_probability(value: float, reference_values: list[float]) -> float:
    return sum(1 for ref in reference_values if ref >= value) / len(reference_values)


def infer_generator(path: Path) -> str:
    name = path.name.lower()
    if "garch" in name:
        return "garch"
    if "sabr" in name:
        return "sabr"
    if "brown" in name:
        return "brown"
    return "unknown"


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def svg_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def save_bar_svg(results: list[dict[str, object]], output_path: Path) -> None:
    ordered = sorted(results, key=lambda row: float(row["mahalanobis_distance"]), reverse=True)
    width, height = 1160, 640
    ml, mr, mt, mb = 90, 30, 70, 190
    pw, ph = width - ml - mr, height - mt - mb
    max_value = max([float(row["mahalanobis_distance"]) for row in ordered] + [1.0])
    gap = 10
    bar_w = (pw - gap * (len(ordered) - 1)) / len(ordered)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="580" y="35" text-anchor="middle" font-size="22" font-family="Arial">Mahalanobis remake2 distance from real-data feature distribution</text>',
        f'<line x1="{ml}" y1="{mt + ph}" x2="{width - mr}" y2="{mt + ph}" stroke="#333"/>',
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + ph}" stroke="#333"/>',
    ]
    for tick in range(6):
        value = max_value * tick / 5
        y = mt + ph - (value / max_value) * ph
        lines.append(f'<line x1="{ml - 5}" y1="{y:.2f}" x2="{width - mr}" y2="{y:.2f}" stroke="#ddd"/>')
        lines.append(f'<text x="{ml - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="12" font-family="Arial">{value:.2f}</text>')
    for idx, row in enumerate(ordered):
        x = ml + idx * (bar_w + gap)
        value = float(row["mahalanobis_distance"])
        bar_h = (value / max_value) * ph
        y = mt + ph - bar_h
        label = svg_escape(str(row["file"]))
        lines.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{bar_h:.2f}" fill="#2a6fbb"/>')
        lines.append(f'<text x="{x + bar_w / 2:.2f}" y="{y - 6:.2f}" text-anchor="middle" font-size="11" font-family="Arial">{value:.2f}</text>')
        lx, ly = x + bar_w / 2, mt + ph + 18
        lines.append(f'<text x="{lx:.2f}" y="{ly:.2f}" transform="rotate(45 {lx:.2f} {ly:.2f})" font-size="12" font-family="Arial">{label}</text>')
    lines.append(f'<text x="22" y="{mt + ph / 2:.2f}" transform="rotate(-90 22 {mt + ph / 2:.2f})" text-anchor="middle" font-size="14" font-family="Arial">Mahalanobis distance</text>')
    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def save_distance_distribution_svg(
    reference_distances: list[dict[str, object]],
    mask_positions: list[dict[str, object]],
    output_path: Path,
) -> None:
    ref_values = [float(row["mahalanobis_distance"]) for row in reference_distances]
    max_value = max(ref_values + [float(row["mahalanobis_distance"]) for row in mask_positions] + [1.0])
    width, height = 1260, 700
    ml, mr, mt, mb = 90, 320, 70, 80
    pw, ph = width - ml - mr, height - mt - mb
    bin_count = 18
    bin_width = max_value / bin_count
    counts = [0 for _ in range(bin_count)]
    for value in ref_values:
        counts[min(int(value / bin_width), bin_count - 1)] += 1
    max_count = max(counts + [1])
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="610" y="35" text-anchor="middle" font-size="22" font-family="Arial">GARCH masks in real-window Mahalanobis remake2 distribution</text>',
        f'<line x1="{ml}" y1="{mt + ph}" x2="{ml + pw}" y2="{mt + ph}" stroke="#333"/>',
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + ph}" stroke="#333"/>',
    ]
    for tick in range(6):
        count = max_count * tick / 5
        y = mt + ph - (count / max_count) * ph
        lines.append(f'<line x1="{ml - 5}" y1="{y:.2f}" x2="{ml + pw}" y2="{y:.2f}" stroke="#ddd"/>')
        lines.append(f'<text x="{ml - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="12" font-family="Arial">{count:.0f}</text>')
    for idx, count in enumerate(counts):
        x = ml + idx * (pw / bin_count)
        bar_h = (count / max_count) * ph
        y = mt + ph - bar_h
        lines.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{pw / bin_count - 3:.2f}" height="{bar_h:.2f}" fill="#d9d9d9" stroke="#999"/>')
    for tick in range(6):
        value = max_value * tick / 5
        x = ml + (value / max_value) * pw
        lines.append(f'<line x1="{x:.2f}" y1="{mt + ph}" x2="{x:.2f}" y2="{mt + ph + 5}" stroke="#333"/>')
        lines.append(f'<text x="{x:.2f}" y="{mt + ph + 22}" text-anchor="middle" font-size="12" font-family="Arial">{value:.1f}</text>')
    for idx, row in enumerate(sorted(mask_positions, key=lambda r: float(r["mahalanobis_distance"]))):
        value = float(row["mahalanobis_distance"])
        x = ml + (value / max_value) * pw
        label_y = mt + 22 + idx * 24
        label = svg_escape(str(row["file"]))
        lines.append(f'<line x1="{x:.2f}" y1="{mt}" x2="{x:.2f}" y2="{mt + ph}" stroke="#2a6fbb" stroke-width="2"/>')
        lines.append(f'<text x="{ml + pw + 18}" y="{label_y:.2f}" font-size="12" font-family="Arial" fill="#2a6fbb">{label}: D={value:.2f}, pct={float(row["reference_percentile"]):.1f}</text>')
    lines.append(f'<text x="{ml + pw / 2:.2f}" y="{height - 20}" text-anchor="middle" font-size="14" font-family="Arial">Mahalanobis distance</text>')
    lines.append(f'<text x="22" y="{mt + ph / 2:.2f}" transform="rotate(-90 22 {mt + ph / 2:.2f})" text-anchor="middle" font-size="14" font-family="Arial">Reference window count</text>')
    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def zscore_color(value: float) -> str:
    clipped = max(min(value, 3.0), -3.0)
    if clipped >= 0:
        intensity = int(255 - (clipped / 3.0) * 120)
        return f"rgb(255,{intensity},{intensity})"
    intensity = int(255 - (abs(clipped) / 3.0) * 120)
    return f"rgb({intensity},{intensity},255)"


def save_heatmap_svg(zscores: list[dict[str, object]], output_path: Path) -> None:
    cell_w, cell_h = 86, 34
    ml, mt = 170, 150
    width = ml + cell_w * len(FEATURE_NAMES) + 40
    height = mt + cell_h * len(zscores) + 40
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.2f}" y="35" text-anchor="middle" font-size="22" font-family="Arial">Remake2 feature z-scores vs real-data reference windows</text>',
    ]
    for col_idx, feature in enumerate(FEATURE_NAMES):
        x, y = ml + col_idx * cell_w + cell_w / 2, mt - 12
        lines.append(f'<text x="{x:.2f}" y="{y:.2f}" transform="rotate(-45 {x:.2f} {y:.2f})" text-anchor="start" font-size="11" font-family="Arial">{feature}</text>')
    for row_idx, row in enumerate(zscores):
        y = mt + row_idx * cell_h
        label = svg_escape(str(row["file"]))
        lines.append(f'<text x="{ml - 8}" y="{y + cell_h * 0.65:.2f}" text-anchor="end" font-size="12" font-family="Arial">{label}</text>')
        for col_idx, feature in enumerate(FEATURE_NAMES):
            x = ml + col_idx * cell_w
            value = float(row[feature])
            lines.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{zscore_color(value)}" stroke="white"/>')
            lines.append(f'<text x="{x + cell_w / 2:.2f}" y="{y + cell_h * 0.65:.2f}" text-anchor="middle" font-size="10" font-family="Arial">{value:.1f}</text>')
    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default="train_data/train_sp500_us10y.csv")
    parser.add_argument("--mask-dir", default="each_mask")
    parser.add_argument("--output-dir", default="mahalanobis_remake1/remake2")
    parser.add_argument("--window-length", type=int, default=1260)
    parser.add_argument("--stride", type=int, default=126)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    features_dir = output_dir / "features"
    results_dir = output_dir / "results"
    figures_dir = output_dir / "figures"
    for directory in [features_dir, results_dir, figures_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    train_sp500, train_dgs10 = read_series(Path(args.train_csv))
    reference_rows = make_reference_features(train_sp500, train_dgs10, args.window_length, args.stride)
    write_csv(
        features_dir / "reference_window_features.csv",
        reference_rows,
        FEATURE_NAMES + ["window_start", "window_end"],
    )

    ref_mean = column_means(reference_rows, FEATURE_NAMES)
    ref_std = [sample_std([float(row[name]) for row in reference_rows]) for name in FEATURE_NAMES]
    covariance = covariance_matrix(reference_rows, FEATURE_NAMES)
    inv_covariance = invert_matrix(covariance)

    reference_distance_rows: list[dict[str, object]] = []
    for row in reference_rows:
        values = [float(row[name]) for name in FEATURE_NAMES]
        reference_distance_rows.append(
            {
                "window_start": row["window_start"],
                "window_end": row["window_end"],
                "mahalanobis_distance": mahalanobis_distance(values, ref_mean, inv_covariance),
            }
        )
    reference_distance_values = [float(row["mahalanobis_distance"]) for row in reference_distance_rows]

    feature_rows: list[dict[str, object]] = []
    result_rows: list[dict[str, object]] = []
    zscore_rows: list[dict[str, object]] = []
    for path in sorted(Path(args.mask_dir).glob("*.csv")):
        sp500, dgs10 = read_series(path)
        features = extract_features(sp500, dgs10)
        distance = mahalanobis_distance([features[name] for name in FEATURE_NAMES], ref_mean, inv_covariance)
        row = {
            "file": path.name,
            "generator": infer_generator(path),
            "n_rows": len(sp500),
            "mahalanobis_distance": distance,
            "reference_percentile": empirical_percentile(distance, reference_distance_values),
            "empirical_upper_tail_probability": empirical_upper_tail_probability(distance, reference_distance_values),
        }
        result_rows.append(row)
        feature_rows.append({"file": path.name, "generator": row["generator"], **features})
        zscores = {
            name: (features[name] - ref_mean[idx]) / ref_std[idx] if ref_std[idx] > 0 else 0.0
            for idx, name in enumerate(FEATURE_NAMES)
        }
        zscore_rows.append({"file": path.name, "generator": row["generator"], **zscores})

    result_rows = sorted(result_rows, key=lambda row: float(row["mahalanobis_distance"]), reverse=True)
    write_csv(
        results_dir / "reference_window_distances.csv",
        reference_distance_rows,
        ["window_start", "window_end", "mahalanobis_distance"],
    )
    write_csv(
        features_dir / "each_mask_features.csv",
        feature_rows,
        ["file", "generator"] + FEATURE_NAMES,
    )
    write_csv(
        results_dir / "mahalanobis_distances.csv",
        result_rows,
        [
            "file",
            "generator",
            "n_rows",
            "mahalanobis_distance",
            "reference_percentile",
            "empirical_upper_tail_probability",
        ],
    )
    write_csv(results_dir / "feature_zscores.csv", zscore_rows, ["file", "generator"] + FEATURE_NAMES)
    save_bar_svg(result_rows, figures_dir / "mahalanobis_distances.svg")
    save_distance_distribution_svg(
        reference_distance_rows,
        result_rows,
        figures_dir / "mask_positions_in_reference_distribution.svg",
    )
    save_heatmap_svg(zscore_rows, figures_dir / "feature_zscores_heatmap.svg")

    selected_features = "\n".join(f"- {name}" for name in FEATURE_NAMES)
    ranking_lines = ["file,generator,n_rows,mahalanobis_distance,reference_percentile,empirical_upper_tail_probability"]
    for row in result_rows:
        ranking_lines.append(
            f'{row["file"]},{row["generator"]},{row["n_rows"]},'
            f'{float(row["mahalanobis_distance"]):.6f},'
            f'{float(row["reference_percentile"]):.2f},'
            f'{float(row["empirical_upper_tail_probability"]):.6f}'
        )

    summary = f"""# Mahalanobis remake2 result

Reference data: {args.train_csv}
Target data: {args.mask_dir}/*.csv
Reference window length: {args.window_length}
Reference stride: {args.stride}
Reference windows: {len(reference_rows)}
Covariance inverse: ordinary sample covariance + Gauss-Jordan inverse

Selected features:
{selected_features}

Joint extreme event definition:
- `joint_abs_q95_move_ratio` is the fraction of rows where both abs(sp500) and abs(DGS10) are at or above their own 95% absolute-move quantile within the same sample/window.

Reference window distance summary:
- min: {min(reference_distance_values):.6f}
- mean: {mean(reference_distance_values):.6f}
- std: {sample_std(reference_distance_values):.6f}
- max: {max(reference_distance_values):.6f}

Ranking by Mahalanobis distance and position in reference distribution:
{chr(10).join(ranking_lines)}
"""
    (results_dir / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
