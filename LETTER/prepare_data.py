import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"user_id", "item_id", "rating", "review"}


def load_csv_rows(csv_path: str) -> List[Dict[str, str]]:
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")
    return df.to_dict(orient="records")


def build_id_maps(rows: List[Dict[str, str]]) -> Tuple[Dict[str, int], Dict[str, int]]:
    user2id: Dict[str, int] = {}
    item2id: Dict[str, int] = {}
    for row in rows:
        user = row["user_id"]
        item = row["item_id"]
        if user not in user2id:
            user2id[user] = len(user2id)
        if item not in item2id:
            item2id[item] = len(item2id)
    return user2id, item2id


def load_indices(npy_path: str) -> np.ndarray:
    indices = np.load(npy_path, allow_pickle=True)
    indices = np.asarray(indices).reshape(-1)
    if indices.dtype.kind not in {"i", "u"}:
        raise ValueError(f"Indices must be integers. Got dtype: {indices.dtype}")
    return indices


def collect_split_rows(rows: List[Dict[str, str]], indices: np.ndarray) -> List[Dict[str, str]]:
    max_index = len(rows) - 1
    subset: List[Dict[str, str]] = []
    for idx in indices.tolist():
        if idx < 0 or idx > max_index:
            raise IndexError(f"Index {idx} out of range 0..{max_index}")
        subset.append(rows[idx])
    return subset


def build_split(
    rows: List[Dict[str, str]],
    indices: np.ndarray,
    user2id: Dict[str, int],
    item2id: Dict[str, int],
) -> Dict[int, List[List[float]]]:
    split: Dict[int, List[List[float]]] = {}
    for idx in indices.tolist():
        row = rows[idx]
        user = user2id[row["user_id"]]
        item = item2id[row["item_id"]]
        rating = float(row["rating"])
        split.setdefault(user, []).append([rating, item])
    return split


def save_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare LETTER train/eval/test json files from CSV + split indices."
    )
    parser.add_argument("--dataset_csv", required=True, help="Path to CSV dataset.")
    parser.add_argument("--train_idx", required=True, help="Path to train indices .npy.")
    parser.add_argument("--eval_idx", required=True, help="Path to eval indices .npy.")
    parser.add_argument("--test_idx", required=True, help="Path to test indices .npy.")
    parser.add_argument("--output_dir", required=True, help="Output directory for json files.")
    args = parser.parse_args()

    train_idx = load_indices(args.train_idx)
    eval_idx = load_indices(args.eval_idx)
    test_idx = load_indices(args.test_idx)

    rows = load_csv_rows(args.dataset_csv)
    all_idx = np.concatenate([train_idx, eval_idx, test_idx])
    split_rows = collect_split_rows(rows, all_idx)
    user2id, item2id = build_id_maps(split_rows)

    train = build_split(rows, train_idx, user2id, item2id)
    eval_ = build_split(rows, eval_idx, user2id, item2id)
    test = build_split(rows, test_idx, user2id, item2id)

    os.makedirs(args.output_dir, exist_ok=True)

    save_json(train, os.path.join(args.output_dir, "train.json"))
    save_json(eval_, os.path.join(args.output_dir, "eval.json"))
    save_json(test, os.path.join(args.output_dir, "test.json"))
    save_json(user2id, os.path.join(args.output_dir, "user2id.json"))
    save_json(item2id, os.path.join(args.output_dir, "item2id.json"))


if __name__ == "__main__":
    main()
