import os
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
    user_doc: torch.LongTensor
    item_doc: torch.LongTensor
    train_dataset: InteractionDataset
    eval_dataset: InteractionDataset
    test_dataset: InteractionDataset
    w2v: np.ndarray


def _build_vocab(train_reviews: List[str], max_vocab: int) -> Dict[str, int]:
    counter = Counter()
    for text in train_reviews:
        counter.update(text.split())

    vocab = {"<unk>": 0}
    for idx, (word, _) in enumerate(counter.most_common(max_vocab), start=1):
        vocab[word] = idx
    return vocab


def _infer_emb_format(path: str) -> str:
    p = path.lower()
    if p.endswith(".npy"):
        return "npy"
    if p.endswith(".bin") or p.endswith(".bin.gz") or p.endswith(".gz"):
        return "word2vec_bin"
    if p.endswith(".txt") or p.endswith(".vec"):
        return "glove_txt"
    return "word2vec_bin"


def _build_w2v_from_word2vec_bin(path: str, vocab: Dict[str, int], word_dim: int) -> np.ndarray:
    try:
        from gensim.models import KeyedVectors
    except Exception as exc:
        raise ImportError("gensim is required to load word2vec .bin/.gz embeddings") from exc

    kv = KeyedVectors.load_word2vec_format(path, binary=True)
    if kv.vector_size != word_dim:
        raise ValueError(
            f"Embedding dim mismatch: file has {kv.vector_size}, but word_dim={word_dim}"
        )

    w2v = np.random.uniform(-1.0, 1.0, size=(len(vocab), word_dim)).astype(np.float32)
    for word, idx in vocab.items():
        if word in kv:
            w2v[idx] = kv[word]
    return w2v


def _build_w2v_from_glove_txt(path: str, vocab: Dict[str, int], word_dim: int) -> np.ndarray:
    w2v = np.random.uniform(-1.0, 1.0, size=(len(vocab), word_dim)).astype(np.float32)
    seen = set()

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) <= word_dim:
                continue
            word = parts[0]
            if word not in vocab:
                continue
            vec = parts[-word_dim:]
            if len(vec) != word_dim:
                continue
            try:
                arr = np.asarray(vec, dtype=np.float32)
            except ValueError:
                continue
            w2v[vocab[word]] = arr
            seen.add(word)
    return w2v


def prepare_data(
    dataset_csv: str,
    train_idx: str,
    eval_idx: str,
    test_idx: str,
    doc_len: int = 500,
    max_vocab: int = 50000,
    word_dim: int = 300,
    pretrained_emb_path: str = "",
    pretrained_emb_format: str = "auto",
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
    item2idx = {i: j for j, i in enumerate(item_values)}

    def split_arrays(split_df: pd.DataFrame):
        u = split_df["user_id"].astype(str).map(user2idx).to_numpy(dtype=np.int64)
        i = split_df["item_id"].astype(str).map(item2idx).to_numpy(dtype=np.int64)
        r = split_df["rating"].astype(np.float32).to_numpy()
        return u, i, r

    tr_u, tr_i, tr_r = split_arrays(train_df)
    va_u, va_i, va_r = split_arrays(eval_df)
    te_u, te_i, te_r = split_arrays(test_df)

    user_reviews = defaultdict(list)
    item_reviews = defaultdict(list)
    train_reviews = []

    for row in train_df.itertuples(index=False):
        u = user2idx[str(getattr(row, "user_id"))]
        i = item2idx[str(getattr(row, "item_id"))]
        review = clean_str(getattr(row, review_col))
        if not review:
            review = "<unk>"
        user_reviews[u].append(review)
        item_reviews[i].append(review)
        train_reviews.append(review)

    vocab = _build_vocab(train_reviews, max_vocab=max_vocab)
    vocab_size = len(vocab)

    user_doc = np.zeros((len(user_values), doc_len), dtype=np.int64)
    item_doc = np.zeros((len(item_values), doc_len), dtype=np.int64)

    for u in range(len(user_values)):
        tokens = " ".join(user_reviews[u] if user_reviews[u] else ["<unk>"]).split()[:doc_len]
        user_doc[u, : len(tokens)] = np.array([vocab.get(t, 0) for t in tokens], dtype=np.int64)

    for i in range(len(item_values)):
        tokens = " ".join(item_reviews[i] if item_reviews[i] else ["<unk>"]).split()[:doc_len]
        item_doc[i, : len(tokens)] = np.array([vocab.get(t, 0) for t in tokens], dtype=np.int64)

    if pretrained_emb_path:
        if not os.path.exists(pretrained_emb_path):
            raise FileNotFoundError(f"Embedding path not found: {pretrained_emb_path}")

        emb_format = _infer_emb_format(pretrained_emb_path) if pretrained_emb_format == "auto" else pretrained_emb_format
        if emb_format == "npy":
            w2v = np.load(pretrained_emb_path).astype(np.float32)
            if w2v.shape[1] != word_dim:
                raise ValueError(
                    f"Embedding dim mismatch: file has {w2v.shape[1]}, but word_dim={word_dim}"
                )
            if w2v.shape[0] < vocab_size:
                pad = np.random.uniform(
                    -1.0, 1.0, size=(vocab_size - w2v.shape[0], word_dim)
                ).astype(np.float32)
                w2v = np.concatenate([w2v, pad], axis=0)
            else:
                w2v = w2v[:vocab_size]
        elif emb_format == "word2vec_bin":
            w2v = _build_w2v_from_word2vec_bin(pretrained_emb_path, vocab, word_dim)
        elif emb_format == "glove_txt":
            w2v = _build_w2v_from_glove_txt(pretrained_emb_path, vocab, word_dim)
        else:
            raise ValueError(f"Unsupported pretrained_emb_format: {emb_format}")
    else:
        w2v = np.random.uniform(-1.0, 1.0, size=(vocab_size, word_dim)).astype(np.float32)

    return PreparedData(
        user_num=len(user_values),
        item_num=len(item_values),
        vocab_size=vocab_size,
        user_doc=torch.from_numpy(user_doc),
        item_doc=torch.from_numpy(item_doc),
        train_dataset=InteractionDataset(tr_u, tr_i, tr_r),
        eval_dataset=InteractionDataset(va_u, va_i, va_r),
        test_dataset=InteractionDataset(te_u, te_i, te_r),
        w2v=w2v,
    )
