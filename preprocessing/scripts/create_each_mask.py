#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def mask_names(headers: list[str]) -> list[str]:
    masks = {
        col.split("_")[0]
        for col in headers
        if col.endswith("_sp500") or col.endswith("_DGS10")
    }

    def sort_key(name: str) -> tuple[int, str]:
        suffix = name.replace("mask", "")
        return (int(suffix), name) if suffix.isdigit() else (10**9, name)

    return sorted(masks, key=sort_key)


def dataset_name(path: Path) -> str:
    name = path.stem
    if name.startswith("mixed_data_"):
        return name[len("mixed_data_") :]
    if name.startswith("mixed_"):
        name = name[len("mixed_") :]
    if name.endswith("_masked"):
        name = name[: -len("_masked")]
    return name


def split_file(input_path: Path, output_dir: Path) -> list[Path]:
    headers, rows = read_csv(input_path)
    dataset = dataset_name(input_path)
    written_paths: list[Path] = []

    for mask in mask_names(headers):
        sp_col = f"{mask}_sp500"
        dgs_col = f"{mask}_DGS10"
        if sp_col not in headers or dgs_col not in headers:
            continue

        output_path = output_dir / f"{mask}_{dataset}.csv"
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["sp500", "DGS10"])
            writer.writeheader()
            for row in rows:
                writer.writerow({"sp500": row[sp_col], "DGS10": row[dgs_col]})

        written_paths.append(output_path)

    return written_paths


def split_directory(input_dir: Path, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths: list[Path] = []

    for input_path in sorted(input_dir.glob("mixed_*_masked.csv")):
        written_paths.extend(split_file(input_path, output_dir))

    if not written_paths:
        raise FileNotFoundError(f"No mixed_*_masked.csv files found in {input_dir}")

    return written_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create each_mask/*.csv files from mixed-format CSV files."
    )
    parser.add_argument(
        "--input-file",
        default=None,
        type=Path,
        help="Single mixed-format CSV file to split. If omitted, --input-dir is scanned.",
    )
    parser.add_argument(
        "--input-dir",
        default=Path("preprocessing/data"),
        type=Path,
        help="Directory containing mixed_*_masked.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("each_mask"),
        type=Path,
        help="Directory where maskN_brown.csv and maskN_sabr.csv files are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.input_file is not None:
        paths = split_file(args.input_file, args.output_dir)
    else:
        paths = split_directory(args.input_dir, args.output_dir)
    print(f"Wrote {len(paths)} files to {args.output_dir}")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
