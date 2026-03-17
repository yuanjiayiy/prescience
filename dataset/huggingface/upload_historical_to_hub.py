"""Upload historical PreScience splits to HuggingFace with schema alignment.

Each subdirectory in ``root_dir`` that matches ``split_regex`` is treated as a
split name (e.g., ``2015_2016``). For each split, the script reads one local
file (JSON or Parquet), aligns columns to the canonical schema split, casts to
the same HuggingFace features, and pushes the resulting DatasetDict.
"""

import argparse
import json
import os
import re
from typing import List

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload historical splits to HuggingFace with canonical schema/features"
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/mmfs1/gscratch/socialrl/jiayiy9/sci4sci/prescience/data/corpus/historical",
        help="Directory containing split subdirectories (e.g., 2015_2016)",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="yuancarrieyjy/PreScience-augmented",
        help="Target HuggingFace dataset repo ID",
    )
    parser.add_argument(
        "--schema_split",
        type=str,
        default="test",
        help="Existing split in repo used as canonical schema/features",
    )
    parser.add_argument(
        "--input_filename",
        type=str,
        default="data.parquet",
        help="Input file name inside each split folder (supports .parquet or .json)",
    )
    parser.add_argument(
        "--json_data_key",
        type=str,
        default="data",
        help="If reading JSON and this key exists, use payload[key] as records",
    )
    parser.add_argument(
        "--split_regex",
        type=str,
        default=r"^\d{4}_\d{4}$",
        help="Regex to select split folder names",
    )
    parser.add_argument(
        "--private_repo",
        action="store_true",
        help="Create/push as private dataset repo",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Optional HF token. If omitted, uses local HuggingFace auth.",
    )
    return parser.parse_args()


def discover_splits(root_dir: str, split_regex: str) -> List[str]:
    pattern = re.compile(split_regex)
    return sorted(
        d
        for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d)) and pattern.match(d)
    )


def load_split_dataframe(file_path: str, json_data_key: str) -> pd.DataFrame:
    if file_path.endswith(".parquet"):
        return pd.read_parquet(file_path)

    if file_path.endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict) and json_data_key in payload:
            payload = payload[json_data_key]
        if not isinstance(payload, list):
            raise ValueError(f"JSON payload must be a list of records, got: {type(payload)}")
        return pd.DataFrame(payload)

    raise ValueError(f"Unsupported input file extension for: {file_path}")


def align_dataframe_to_schema(df: pd.DataFrame, schema_columns: List[str]) -> pd.DataFrame:
    for column in schema_columns:
        if column not in df.columns:
            df[column] = pd.NA
    return df[schema_columns]


def main() -> None:
    args = parse_args()

    api = HfApi(token=args.token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private_repo,
        exist_ok=True,
    )

    dataset_dict: DatasetDict = load_dataset(args.repo_id)
    if args.schema_split not in dataset_dict:
        raise ValueError(
            f"schema_split '{args.schema_split}' not found in dataset. Available: {list(dataset_dict.keys())}"
        )

    canonical_features = dataset_dict[args.schema_split].features
    canonical_columns = list(canonical_features.keys())

    print(f"Canonical schema split: {args.schema_split}")
    print(f"Canonical columns ({len(canonical_columns)}): {canonical_columns}")

    split_names = discover_splits(args.root_dir, args.split_regex)
    print(f"Found {len(split_names)} candidate splits under {args.root_dir}")

    for split_name in split_names:
        split_file = os.path.join(args.root_dir, split_name, args.input_filename)
        print(f"\n[PROCESS] {split_name}")

        if not os.path.exists(split_file):
            print(f"[SKIP] {split_name}: missing {args.input_filename}")
            continue

        print(f"[LOAD] {split_file}")
        split_df = load_split_dataframe(split_file, args.json_data_key)
        print(f"Original columns ({len(split_df.columns)}): {list(split_df.columns)}")

        split_df = align_dataframe_to_schema(split_df, canonical_columns)
        split_ds = Dataset.from_pandas(split_df, preserve_index=False).cast(canonical_features)

        dataset_dict[split_name] = split_ds
        print(f"[READY] {split_name}: rows={len(split_ds)}")

    dataset_dict.push_to_hub(args.repo_id, private=args.private_repo, token=args.token)
    print("\nDone.")


if __name__ == "__main__":
    main()
