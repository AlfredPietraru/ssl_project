import argparse
import os
from pathlib import Path
from typing import TextIO

os.environ.setdefault("MPLCONFIGDIR", str(Path("tmp/matplotlib").resolve()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import pandas as pd
from wildlife_datasets.datasets import AnimalCLEF2026


def clean_string_series(values: pd.Series, missing_value: str = "unknown") -> pd.Series:
    series = pd.Series(values, dtype="object")
    series = series.where(series.notna(), missing_value)
    series = series.astype(str).str.strip()
    return series.mask(series.eq("") | series.str.lower().isin({"nan", "none"}), missing_value)


def infer_species_from_text(row: pd.Series) -> str:
    for column in ("identity", "dataset", "path", "image_id"):
        value = str(row.get(column, "")).lower()
        if "salamander" in value:
            return "salamander"
        if "lynx" in value:
            return "lynx"
        if "turtle" in value:
            return "loggerhead turtle"
        if "lizard" in value:
            return "lizard"
    return "unknown"


def fill_missing_species(df: pd.DataFrame) -> pd.Series:
    species = clean_string_series(df["species"])
    missing_species = species.eq("unknown")
    if missing_species.any():
        species.loc[missing_species] = df.loc[missing_species].apply(infer_species_from_text, axis=1)
    return species


def load_metadata(root: Path) -> pd.DataFrame:
    dataset = AnimalCLEF2026(
        str(root),
        transform=None,
        load_label=True,
        factorize_label=True,
        check_files=False,
    )
    df = dataset.df.copy()

    required_columns = {"split", "species", "identity"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset metadata is missing required columns: {sorted(missing_columns)}")

    df["split"] = clean_string_series(df["split"])
    df["identity"] = clean_string_series(df["identity"])
    df["species"] = fill_missing_species(df)
    return df


def identity_counts_by_species(df: pd.DataFrame, split: str) -> dict[str, pd.Series]:
    split_df = df[df["split"] == split]
    grouped = {}
    for species, species_df in split_df.groupby("species", sort=True):
        grouped[species] = species_df["identity"].value_counts().sort_index()
    return grouped


def format_identity_counts(counts: pd.Series) -> list[str]:
    return [f"{identity} ({count})" for identity, count in counts.items()]


def write_line(output: TextIO, text: str = "") -> None:
    output.write(f"{text}\n")


def write_split_report(df: pd.DataFrame, split: str, output: TextIO) -> None:
    split_df = df[df["split"] == split]
    grouped = identity_counts_by_species(df, split)

    write_line(output)
    write_line(output, split.upper())
    write_line(output, "=" * len(split))
    write_line(output, f"Samples: {len(split_df)}")
    write_line(output, f"Species/classes: {len(grouped)}")
    write_line(output, f"Unique identities: {split_df['identity'].nunique()}")

    for species, counts in grouped.items():
        write_line(output)
        write_line(output, f"[{species}] {len(counts)} identities, {int(counts.sum())} samples")
        for identity in format_identity_counts(counts):
            write_line(output, f"  - {identity}")


def write_comparison_report(df: pd.DataFrame, output: TextIO) -> None:
    train = identity_counts_by_species(df, "train")
    test = identity_counts_by_species(df, "test")
    species_names = sorted(set(train) | set(test))

    write_line(output)
    write_line(output, "COMPARISON")
    write_line(output, "==========")
    for species in species_names:
        train_ids = set(train.get(species, pd.Series(dtype=int)).index)
        test_ids = set(test.get(species, pd.Series(dtype=int)).index)

        only_train = sorted(train_ids - test_ids)
        only_test = sorted(test_ids - train_ids)
        both = sorted(train_ids & test_ids)

        write_line(output)
        write_line(output, f"[{species}]")
        write_line(output, f"  train identities: {len(train_ids)}")
        write_line(output, f"  test identities: {len(test_ids)}")
        write_line(output, f"  in both splits: {len(both)}")
        write_line(output, f"  only train: {len(only_train)}")
        write_line(output, f"  only test: {len(only_test)}")

        if both:
            write_line(output, "  shared:")
            for identity in both:
                write_line(output, f"    - {identity}")
        if only_train:
            write_line(output, "  train-only:")
            for identity in only_train:
                write_line(output, f"    - {identity}")
        if only_test:
            write_line(output, "  test-only:")
            for identity in only_test:
                write_line(output, f"    - {identity}")


def write_report(df: pd.DataFrame, mode: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        if mode in {"splits", "all"}:
            write_split_report(df, "train", output)
            write_split_report(df, "test", output)
        if mode in {"comparison", "all"}:
            write_comparison_report(df, output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print AnimalCLEF2026 identity names by species/class for train and test splits."
    )
    parser.add_argument("--root", default="data", help="Dataset root directory.")
    parser.add_argument(
        "--mode",
        choices=("splits", "comparison", "all"),
        default="all",
        help="Report style to print.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/dataset_identity_analysis.txt",
        help="Path where the report will be written.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root '{root}' does not exist.")

    df = load_metadata(root)
    output_path = Path(args.output)
    write_report(df, args.mode, output_path)

    print(f"Wrote dataset identity analysis to {output_path}")


if __name__ == "__main__":
    main()
