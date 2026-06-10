#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


FEATURE_NAMES = [
    "sp500_std",
    "sp500_q01",
    "sp500_q05",
    "sp500_q95",
    "sp500_q99",
    "sp500_abs_autocorr_lag1",
    "sp500_abs_autocorr_lag5",
    "sp500_abs_autocorr_lag20",
    "dgs10_std",
    "dgs10_q01",
    "dgs10_q05",
    "dgs10_q95",
    "dgs10_q99",
    "dgs10_abs_autocorr_lag1",
    "dgs10_abs_autocorr_lag5",
    "dgs10_abs_autocorr_lag20",
    "cross_corr",
    "rolling_corr_std_60",
    "corr_down_sp500_q05",
    "corr_up_sp500_q95",
]


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def column_means(rows: list[dict[str, str]], names: list[str]) -> list[float]:
    return [mean([float(row[name]) for row in rows]) for name in names]


def covariance_matrix(rows: list[dict[str, str]], names: list[str]) -> list[list[float]]:
    n = len(rows)
    mus = column_means(rows, names)
    matrix: list[list[float]] = []
    for i, name_i in enumerate(names):
        row_values: list[float] = []
        for j, name_j in enumerate(names):
            value = sum((float(row[name_i]) - mus[i]) * (float(row[name_j]) - mus[j]) for row in rows)
            row_values.append(value / (n - 1))
        matrix.append(row_values)
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


def t_squared(values: list[float], mus: list[float], inv_cov: list[list[float]]) -> float:
    diff = [x - mu for x, mu in zip(values, mus)]
    transformed = mat_vec_mul(inv_cov, diff)
    return max(sum(a * b for a, b in zip(diff, transformed)), 0.0)


def beta_continued_fraction(a: float, b: float, x: float) -> float:
    max_iter = 200
    eps = 3.0e-14
    fpmin = 1.0e-300

    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d

    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_bt = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    bt = math.exp(log_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * beta_continued_fraction(a, b, x) / a
    return 1.0 - bt * beta_continued_fraction(b, a, 1.0 - x) / b


def f_cdf(x: float, dfn: int, dfd: int) -> float:
    if x <= 0.0:
        return 0.0
    z = (dfn * x) / (dfn * x + dfd)
    return regularized_incomplete_beta(dfn / 2.0, dfd / 2.0, z)


def f_sf(x: float, dfn: int, dfd: int) -> float:
    return max(0.0, min(1.0, 1.0 - f_cdf(x, dfn, dfd)))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def svg_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def save_pvalue_svg(results: list[dict[str, object]], output_path: Path) -> None:
    ordered = sorted(results, key=lambda row: float(row["p_value"]))
    width, height = 1100, 620
    ml, mr, mt, mb = 90, 30, 70, 180
    pw, ph = width - ml - mr, height - mt - mb
    gap = 12
    bar_w = (pw - gap * (len(ordered) - 1)) / len(ordered)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="550" y="35" text-anchor="middle" font-size="22" font-family="Arial">Hotelling T² p-values vs real-data reference windows</text>',
        f'<line x1="{ml}" y1="{mt + ph}" x2="{width - mr}" y2="{mt + ph}" stroke="#333"/>',
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + ph}" stroke="#333"/>',
    ]
    for tick in range(6):
        value = tick / 5
        y = mt + ph - value * ph
        lines.append(f'<line x1="{ml - 5}" y1="{y:.2f}" x2="{width - mr}" y2="{y:.2f}" stroke="#ddd"/>')
        lines.append(f'<text x="{ml - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="12" font-family="Arial">{value:.1f}</text>')
    alpha_y = mt + ph - 0.05 * ph
    lines.append(f'<line x1="{ml}" y1="{alpha_y:.2f}" x2="{width - mr}" y2="{alpha_y:.2f}" stroke="#d62728" stroke-dasharray="6 4"/>')
    lines.append(f'<text x="{width - mr - 4}" y="{alpha_y - 6:.2f}" text-anchor="end" font-size="12" font-family="Arial" fill="#d62728">p=0.05</text>')

    for idx, row in enumerate(ordered):
        x = ml + idx * (bar_w + gap)
        p_value = float(row["p_value"])
        bar_h = p_value * ph
        y = mt + ph - bar_h
        color = "#9467bd" if row["generator"] == "sabr" else "#1f77b4"
        label = svg_escape(str(row["file"]))
        lines.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{bar_h:.2f}" fill="{color}"/>')
        lines.append(f'<text x="{x + bar_w / 2:.2f}" y="{max(y - 6, mt + 12):.2f}" text-anchor="middle" font-size="11" font-family="Arial">{p_value:.3f}</text>')
        lx, ly = x + bar_w / 2, mt + ph + 18
        lines.append(f'<text x="{lx:.2f}" y="{ly:.2f}" transform="rotate(45 {lx:.2f} {ly:.2f})" font-size="12" font-family="Arial">{label}</text>')
    lines.append(f'<text x="22" y="{mt + ph / 2:.2f}" transform="rotate(-90 22 {mt + ph / 2:.2f})" text-anchor="middle" font-size="14" font-family="Arial">p-value</text>')
    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-features", default="mahalanobis_remake1/features/reference_window_features.csv")
    parser.add_argument("--mask-features", default="mahalanobis_remake1/features/each_mask_features.csv")
    parser.add_argument("--output-dir", default="mahalanobis_remake1/hotelling_t2")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    results_dir = output_dir / "results"
    figures_dir = output_dir / "figures"
    logs_dir = output_dir / "logs"
    for directory in [results_dir, figures_dir, logs_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    reference_rows = read_csv_dicts(Path(args.reference_features))
    mask_rows = read_csv_dicts(Path(args.mask_features))
    n = len(reference_rows)
    p = len(FEATURE_NAMES)
    if n <= p:
        raise ValueError(f"need n > p for Hotelling T2, got n={n}, p={p}")

    ref_mean = column_means(reference_rows, FEATURE_NAMES)
    covariance = covariance_matrix(reference_rows, FEATURE_NAMES)
    inv_covariance = invert_matrix(covariance)

    dfn = p
    dfd = n - p
    scale = n * (n - p) / (p * (n + 1) * (n - 1))

    results: list[dict[str, object]] = []
    for row in mask_rows:
        values = [float(row[name]) for name in FEATURE_NAMES]
        t2 = t_squared(values, ref_mean, inv_covariance)
        f_stat = scale * t2
        p_value = f_sf(f_stat, dfn, dfd)
        results.append(
            {
                "file": row["file"],
                "generator": row["generator"],
                "t_squared": t2,
                "f_statistic": f_stat,
                "dfn": dfn,
                "dfd": dfd,
                "p_value": p_value,
                "reject_0.05": p_value < 0.05,
                "reject_0.01": p_value < 0.01,
            }
        )

    results = sorted(results, key=lambda item: float(item["p_value"]))
    write_csv(
        results_dir / "hotelling_t2_results.csv",
        results,
        ["file", "generator", "t_squared", "f_statistic", "dfn", "dfd", "p_value", "reject_0.05", "reject_0.01"],
    )
    save_pvalue_svg(results, figures_dir / "hotelling_t2_pvalues.svg")

    ranking_lines = ["file,generator,t_squared,f_statistic,p_value,reject_0.05,reject_0.01"]
    for row in results:
        ranking_lines.append(
            f'{row["file"]},{row["generator"]},{float(row["t_squared"]):.6f},'
            f'{float(row["f_statistic"]):.6f},{float(row["p_value"]):.8f},'
            f'{row["reject_0.05"]},{row["reject_0.01"]}'
        )
    summary = f"""# Hotelling T2 result

Reference features: {args.reference_features}
Target features: {args.mask_features}
Reference windows n: {n}
Feature dimension p: {p}
F distribution degrees of freedom: ({dfn}, {dfd})
Phase-II scaling: F = n * (n - p) / [p * (n + 1) * (n - 1)] * T2

Ranking by p-value:
{chr(10).join(ranking_lines)}

Interpretation:
- Smaller p-value means the feature vector is statistically farther from the real-data reference distribution.
- p < 0.05 is treated as outside the 95% Hotelling T2 prediction region.
- This uses the same ordinary sample mean and ordinary sample covariance as mahalanobis_remake1.
"""
    (results_dir / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
