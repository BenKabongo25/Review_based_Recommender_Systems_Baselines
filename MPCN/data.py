import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def clean_str(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]", " ", str(text))
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip().lower()


def _load_indices(path: str) -> np.ndarray:
    idx = np.load(path, allow_pickle=True)
    idx = np.asarray(idx).reshape(-1)
    if idx.dtype.kind not in {"i", "u"}:
        raise ValueError(f"Indices must be integer dtype, got {idx.dtype}")
    if idx.size == 0:
        raise ValueError(f"Empty split index file: {path}")
    return idx.astype(np.int64)


class InteractionDataset(Dataset):
    def __init__(self, uids: np.ndarray, iids: np.ndarray, ratings: np.ndarray):
        self.uids = torch.from_numpy(uids.astype(np.int64))
        self.iids = torch.from_numpy(iids.astype(np.int64))
        self.ratings = torch.from_numpy(ratings.astype(np.float32))

    def __len__(self):
        return self.ratings.shape[0]

    def __getitem__(self, idx):
        return self.uids[idx], self.iids[idx], self.ratings[idx]


@dataclass
class PreparedData:
    user_num: int
    item_num: int
    vocab_size: int
    users_review_list: torch.LongTensor
    items_review_list: torch.LongTensor
    train_dataset: InteractionDataset
    eval_dataset: InteractionDataset
    test_dataset: InteractionDataset


def _build_vocab(train_reviews: List[str], max_vocab: int) -> Dict[str, int]:
    counter = Counter()
    for r in train_reviews:
        counter.update(r.split())
    vocab = {"<pad>": 0, "<unk>": 1}
    for i, (w, _) in enumerate(counter.most_common(max_vocab), start=2):
        vocab[w] = i
    return vocab


def _encode_tokens(tokens: List[str], vocab: Dict[str, int], max_len: int) -> List[int]:
    ids = [vocab.get(w, 1) for w in tokens[:max_len]]
    if len(ids) < max_len:
        ids.extend([0] * (max_len - len(ids)))
    return ids


def prepare_data(
    dataset_csv: str,
    train_idx: str,
    eval_idx: str,
    test_idx: str,
    max_vocab: int = 50000,
    review_num: int = 20,
    review_len: int = 30,
) -> PreparedData:
    df = pd.read_csv(dataset_csv)
    required_cols = {"user_id", "item_id", "rating"}
    if not required_cols.issubset(df.columns):
        raise ValueError("CSV must contain user_id, item_id, rating columns")

    review_col = "review" if "review" in df.columns else "review_text"
    if review_col not in df.columns:
        raise ValueError("CSV must contain `review` or `review_text` column")

    tr_idx = _load_indices(train_idx)
    va_idx = _load_indices(eval_idx)
    te_idx = _load_indices(test_idx)

    for name, ids in (("train", tr_idx), ("eval", va_idx), ("test", te_idx)):
        if ids.min() < 0 or ids.max() >= len(df):
            raise IndexError(f"{name} indices out of bounds for dataset size {len(df)}")

    train_df = df.iloc[tr_idx].copy()
    eval_df = df.iloc[va_idx].copy()
    test_df = df.iloc[te_idx].copy()

    all_df = pd.concat([train_df, eval_df, test_df], ignore_index=True)
    user_values = all_df["user_id"].astype(str).drop_duplicates().tolist()
    item_values = all_df["item_id"].astype(str).drop_duplicates().tolist()
    user2idx = {u: i for i, u in enumerate(user_values)}
    item2idx = {it: i for i, it in enumerate(item_values)}

    def split_arrays(split_df: pd.DataFrame):
        u = split_df["user_id"].astype(str).map(user2idx).to_numpy(dtype=np.int64)
        i = split_df["item_id"].astype(str).map(item2idx).to_numpy(dtype=np.int64)
        r = split_df["rating"].astype(np.float32).to_numpy()
        return u, i, r

    tr_u, tr_i, tr_r = split_arrays(train_df)
    va_u, va_i, va_r = split_arrays(eval_df)
    te_u, te_i, te_r = split_arrays(test_df)

    user_reviews_dict: Dict[int, List[str]] = defaultdict(list)
    item_reviews_dict: Dict[int, List[str]] = defaultdict(list)
    clean_train_reviews: List[str] = []

    for row in train_df.itertuples(index=False):
        u = user2idx[str(getattr(row, "user_id"))]
        i = item2idx[str(getattr(row, "item_id"))]
        review = clean_str(getattr(row, review_col))
        if not review:
            review = "<unk>"
        user_reviews_dict[u].append(review)
        item_reviews_dict[i].append(review)
        clean_train_reviews.append(review)

    vocab = _build_vocab(clean_train_reviews, max_vocab=max_vocab)

    user_reviews = np.zeros((len(user_values), review_num, review_len), dtype=np.int64)
    item_reviews = np.zeros((len(item_values), review_num, review_len), dtype=np.int64)

    for u in range(len(user_values)):
        reviews = user_reviews_dict[u] if user_reviews_dict[u] else ["<unk>"]
        for ridx, txt in enumerate(reviews[:review_num]):
            user_reviews[u, ridx] = np.array(_encode_tokens(txt.split(), vocab, review_len), dtype=np.int64)

    for i in range(len(item_values)):
        reviews = item_reviews_dict[i] if item_reviews_dict[i] else ["<unk>"]
        for ridx, txt in enumerate(reviews[:review_num]):
            item_reviews[i, ridx] = np.array(_encode_tokens(txt.split(), vocab, review_len), dtype=np.int64)

    return PreparedData(
        user_num=len(user_values),
        item_num=len(item_values),
        vocab_size=len(vocab),
        users_review_list=torch.from_numpy(user_reviews),
        items_review_list=torch.from_numpy(item_reviews),
        train_dataset=InteractionDataset(tr_u, tr_i, tr_r),
        eval_dataset=InteractionDataset(va_u, va_i, va_r),
        test_dataset=InteractionDataset(te_u, te_i, te_r),
    )
