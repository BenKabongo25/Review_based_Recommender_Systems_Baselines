import re
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


REVIEW_PERCENTILE = 0.85


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
    return idx


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
    user2itemid_list: torch.LongTensor
    item2userid_list: torch.LongTensor
    user_doc: torch.LongTensor
    item_doc: torch.LongTensor
    train_dataset: InteractionDataset
    eval_dataset: InteractionDataset
    test_dataset: InteractionDataset
    w2v: np.ndarray


def _compute_percentile_length(lengths: List[int], p: float, minimum: int = 1) -> int:
    if not lengths:
        return minimum
    arr = np.sort(np.array(lengths, dtype=np.int64))
    pos = int(max(1, np.ceil(p * len(arr)))) - 1
    return int(max(minimum, arr[pos]))


def _build_vocab(train_reviews: List[str], max_vocab: int) -> Dict[str, int]:
    counter = Counter()
    for r in train_reviews:
        counter.update(r.split())
    most_common = counter.most_common(max_vocab)
    vocab = {"<unk>": 0}
    for i, (w, _) in enumerate(most_common, start=1):
        vocab[w] = i
    return vocab


def _encode_tokens(tokens: List[str], vocab: Dict[str, int], max_len: int) -> List[int]:
    ids = [vocab.get(w, 0) for w in tokens]
    if len(ids) < max_len:
        ids = ids + [0] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids


def prepare_data(
    dataset_csv: str,
    train_idx: str,
    eval_idx: str,
    test_idx: str,
    doc_len: int = 500,
    max_vocab: int = 50000,
    p_review: float = REVIEW_PERCENTILE,
    word_dim: int = 300,
    w2v_path: str = "",
) -> PreparedData:
    df = pd.read_csv(dataset_csv)
    if not {"user_id", "item_id", "rating"}.issubset(df.columns):
        raise ValueError("CSV must contain user_id, item_id, rating columns")
    review_col = "review" if "review" in df.columns else "review_text"
    if review_col not in df.columns:
        raise ValueError("CSV must contain review or review_text column")

    tr = _load_indices(train_idx)
    va = _load_indices(eval_idx)
    te = _load_indices(test_idx)
    for name, ids in (("train", tr), ("eval", va), ("test", te)):
        if ids.min() < 0 or ids.max() >= len(df):
            raise IndexError(f"{name} indices out of bounds for dataset size {len(df)}")

    train_df = df.iloc[tr].copy()
    eval_df = df.iloc[va].copy()
    test_df = df.iloc[te].copy()

    all_df = pd.concat([train_df, eval_df, test_df], ignore_index=True)
    user_ids = all_df["user_id"].astype(str).drop_duplicates().tolist()
    item_ids = all_df["item_id"].astype(str).drop_duplicates().tolist()
    user2idx = {u: i for i, u in enumerate(user_ids)}
    item2idx = {m: i for i, m in enumerate(item_ids)}

    user_num = len(user_ids)
    item_num = len(item_ids)

    def _split_to_arrays(split_df: pd.DataFrame):
        uids = split_df["user_id"].astype(str).map(user2idx).to_numpy(dtype=np.int64)
        iids = split_df["item_id"].astype(str).map(item2idx).to_numpy(dtype=np.int64)
        ratings = split_df["rating"].astype(np.float32).to_numpy()
        return uids, iids, ratings

    tr_u, tr_i, tr_r = _split_to_arrays(train_df)
    va_u, va_i, va_r = _split_to_arrays(eval_df)
    te_u, te_i, te_r = _split_to_arrays(test_df)

    # Build review dictionaries from train split only.
    user_reviews_dict: Dict[int, List[str]] = defaultdict(list)
    item_reviews_dict: Dict[int, List[str]] = defaultdict(list)
    user_item_ids: Dict[int, List[int]] = defaultdict(list)
    item_user_ids: Dict[int, List[int]] = defaultdict(list)
    clean_train_reviews: List[str] = []

    for _, row in train_df.iterrows():
        u = user2idx[str(row["user_id"])]
        i = item2idx[str(row["item_id"])]
        review = clean_str(row[review_col])
        if not review:
            review = "<unk>"
        user_reviews_dict[u].append(review)
        item_reviews_dict[i].append(review)
        user_item_ids[u].append(i)
        item_user_ids[i].append(u)
        clean_train_reviews.append(review)

    vocab = _build_vocab(clean_train_reviews, max_vocab=max_vocab)

    # Determine truncation lengths from train distributions.
    user_review_counts = [len(user_reviews_dict[u]) for u in range(user_num)]
    item_review_counts = [len(item_reviews_dict[i]) for i in range(item_num)]
    sent_lens = []
    for r in clean_train_reviews:
        sent_lens.append(max(1, len(r.split())))

    u_max_r = _compute_percentile_length(user_review_counts, p_review, minimum=1)
    i_max_r = _compute_percentile_length(item_review_counts, p_review, minimum=1)
    max_sent_len = _compute_percentile_length(sent_lens, p_review, minimum=1)

    user_reviews_arr = np.zeros((user_num, u_max_r, max_sent_len), dtype=np.int64)
    item_reviews_arr = np.zeros((item_num, i_max_r, max_sent_len), dtype=np.int64)
    user_item2id = np.full((user_num, u_max_r), fill_value=item_num + 1, dtype=np.int64)
    item_user2id = np.full((item_num, i_max_r), fill_value=user_num + 1, dtype=np.int64)
    user_doc_arr = np.zeros((user_num, doc_len), dtype=np.int64)
    item_doc_arr = np.zeros((item_num, doc_len), dtype=np.int64)

    def _encode_review_list(reviews: List[str], max_r: int) -> np.ndarray:
        rows = np.zeros((max_r, max_sent_len), dtype=np.int64)
        for ridx, txt in enumerate(reviews[:max_r]):
            rows[ridx] = np.array(_encode_tokens(txt.split(), vocab, max_sent_len), dtype=np.int64)
        return rows

    for u in range(user_num):
        reviews = user_reviews_dict[u] if user_reviews_dict[u] else ["<unk>"]
        user_reviews_arr[u] = _encode_review_list(reviews, u_max_r)
        ids = user_item_ids[u][:u_max_r]
        user_item2id[u, : len(ids)] = np.array(ids, dtype=np.int64)
        doc_tokens = " ".join(reviews).split()[:doc_len]
        user_doc_arr[u, : len(doc_tokens)] = np.array([vocab.get(w, 0) for w in doc_tokens], dtype=np.int64)

    for i in range(item_num):
        reviews = item_reviews_dict[i] if item_reviews_dict[i] else ["<unk>"]
        item_reviews_arr[i] = _encode_review_list(reviews, i_max_r)
        ids = item_user_ids[i][:i_max_r]
        item_user2id[i, : len(ids)] = np.array(ids, dtype=np.int64)
        doc_tokens = " ".join(reviews).split()[:doc_len]
        item_doc_arr[i, : len(doc_tokens)] = np.array([vocab.get(w, 0) for w in doc_tokens], dtype=np.int64)

    vocab_size = len(vocab)
    if w2v_path and os.path.exists(w2v_path):
        w2v = np.load(w2v_path)
        if w2v.shape[0] < vocab_size:
            pad = np.random.uniform(-1.0, 1.0, size=(vocab_size - w2v.shape[0], w2v.shape[1])).astype(np.float32)
            w2v = np.concatenate([w2v.astype(np.float32), pad], axis=0)
        else:
            w2v = w2v[:vocab_size].astype(np.float32)
    else:
        w2v = np.random.uniform(-1.0, 1.0, size=(vocab_size, word_dim)).astype(np.float32)

    return PreparedData(
        user_num=user_num,
        item_num=item_num,
        vocab_size=vocab_size,
        users_review_list=torch.from_numpy(user_reviews_arr),
        items_review_list=torch.from_numpy(item_reviews_arr),
        user2itemid_list=torch.from_numpy(user_item2id),
        item2userid_list=torch.from_numpy(item_user2id),
        user_doc=torch.from_numpy(user_doc_arr),
        item_doc=torch.from_numpy(item_doc_arr),
        train_dataset=InteractionDataset(tr_u, tr_i, tr_r),
        eval_dataset=InteractionDataset(va_u, va_i, va_r),
        test_dataset=InteractionDataset(te_u, te_i, te_r),
        w2v=w2v,
    )
