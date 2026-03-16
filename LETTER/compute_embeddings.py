import argparse
import json
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def load_indices(npy_path: str) -> np.ndarray:
    indices = np.load(npy_path, allow_pickle=True)
    indices = np.asarray(indices).reshape(-1)
    if indices.dtype.kind not in {"i", "u"}:
        raise ValueError(f"Indices must be integers. Got dtype: {indices.dtype}")
    return indices


def load_id_map(path: str) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): int(v) for k, v in data.items()}


def resolve_id(id_map: Dict[str, int], raw_id) -> int:
    key = str(raw_id)
    if key not in id_map:
        raise KeyError(f"ID {raw_id} not found in mapping.")
    return id_map[key]


def init_accumulators(num_entities: int, dim: int) -> Tuple[np.ndarray, np.ndarray]:
    sums = np.zeros((num_entities, dim), dtype=np.float32)
    counts = np.zeros((num_entities,), dtype=np.int64)
    return sums, counts


def finalize_embeddings(sums: np.ndarray, counts: np.ndarray) -> np.ndarray:
    embeddings = np.zeros_like(sums)
    nonzero = counts > 0
    if np.any(nonzero):
        embeddings[nonzero] = sums[nonzero] / counts[nonzero][:, None]
    return embeddings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute user/item embeddings from train interactions."
    )
    parser.add_argument("--dataset_csv", required=True, help="Path to CSV dataset.")
    parser.add_argument("--train_idx", required=True, help="Path to train indices .npy.")
    parser.add_argument("--user2id", required=True, help="Path to user2id.json.")
    parser.add_argument("--item2id", required=True, help="Path to item2id.json.")
    parser.add_argument("--output_dir", required=True, help="Output directory for embeddings.")
    parser.add_argument(
        "--model_name",
        default="sentence-transformers/msmarco-bert-base-dot-v5",
        help="SentenceTransformer model name or path.",
    )
    parser.add_argument(
        "--rating_threshold",
        default=3.0,
        type=float,
        help="Ratings > threshold are likes; <= are dislikes.",
    )
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--device", default=None, help="Device for SentenceTransformer.")
    args = parser.parse_args()

    df = pd.read_csv(args.dataset_csv)
    train_idx = load_indices(args.train_idx)
    train_df = df.iloc[train_idx]

    user2id = load_id_map(args.user2id)
    item2id = load_id_map(args.item2id)
    num_users = len(user2id)
    num_items = len(item2id)

    reviews = train_df["review"].astype(str).tolist()
    user_raw = train_df["user_id"].tolist()
    item_raw = train_df["item_id"].tolist()
    ratings = train_df["rating"].astype(float).to_numpy()

    model = SentenceTransformer(args.model_name, device=args.device)
    review_emb = model.encode(
        reviews, batch_size=args.batch_size, show_progress_bar=True
    )
    review_emb = np.asarray(review_emb, dtype=np.float32)
    dim = review_emb.shape[1]

    user_sum, user_cnt = init_accumulators(num_users, dim)
    item_sum, item_cnt = init_accumulators(num_items, dim)
    user_like_sum, user_like_cnt = init_accumulators(num_users, dim)
    user_dis_sum, user_dis_cnt = init_accumulators(num_users, dim)
    item_like_sum, item_like_cnt = init_accumulators(num_items, dim)
    item_dis_sum, item_dis_cnt = init_accumulators(num_items, dim)

    for idx in range(len(train_df)):
        u_id = resolve_id(user2id, user_raw[idx])
        i_id = resolve_id(item2id, item_raw[idx])
        emb = review_emb[idx]

        user_sum[u_id] += emb
        user_cnt[u_id] += 1
        item_sum[i_id] += emb
        item_cnt[i_id] += 1

        if ratings[idx] > args.rating_threshold:
            user_like_sum[u_id] += emb
            user_like_cnt[u_id] += 1
            item_like_sum[i_id] += emb
            item_like_cnt[i_id] += 1
        else:
            user_dis_sum[u_id] += emb
            user_dis_cnt[u_id] += 1
            item_dis_sum[i_id] += emb
            item_dis_cnt[i_id] += 1

    user_global = finalize_embeddings(user_sum, user_cnt)
    item_global = finalize_embeddings(item_sum, item_cnt)
    user_like = finalize_embeddings(user_like_sum, user_like_cnt)
    user_dislike = finalize_embeddings(user_dis_sum, user_dis_cnt)
    item_like = finalize_embeddings(item_like_sum, item_like_cnt)
    item_dislike = finalize_embeddings(item_dis_sum, item_dis_cnt)

    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, "user_global.npy"), user_global)
    np.save(os.path.join(args.output_dir, "user_like.npy"), user_like)
    np.save(os.path.join(args.output_dir, "user_dislike.npy"), user_dislike)
    np.save(os.path.join(args.output_dir, "item_global.npy"), item_global)
    np.save(os.path.join(args.output_dir, "item_like.npy"), item_like)
    np.save(os.path.join(args.output_dir, "item_dislike.npy"), item_dislike)


if __name__ == "__main__":
    main()
