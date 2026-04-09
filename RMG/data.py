import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


WORD_RE = re.compile(r"[^A-Za-z0-9]+")
SENT_SPLIT_RE = re.compile(r"[.!?]+")


def _load_indices(path: str) -> np.ndarray:
    idx = np.load(path, allow_pickle=True)
    idx = np.asarray(idx).reshape(-1)
    if idx.dtype.kind not in {"i", "u"}:
        raise ValueError(f"Indices must be integer dtype, got {idx.dtype}")
    if idx.size == 0:
        raise ValueError(f"Empty split index file: {path}")
    return idx.astype(np.int64)


def _tokenize_sentence(text: str) -> List[str]:
    text = WORD_RE.sub(" ", str(text)).strip().lower()
    if not text:
        return []
    return text.split()


def _split_review(review: str) -> List[List[str]]:
    pieces = SENT_SPLIT_RE.split(str(review))
    sents = []
    for p in pieces:
        tokens = _tokenize_sentence(p)
        if tokens:
            sents.append(tokens)
    if not sents:
        # Keep one fallback sentence to avoid fully empty reviews.
        sents = [["<unk>"]]
    return sents


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
    user_docs: torch.LongTensor
    item_docs: torch.LongTensor
    user_neighbors: torch.LongTensor
    item_neighbors: torch.LongTensor
    user_item_user: torch.LongTensor
    item_user_item: torch.LongTensor
    train_dataset: InteractionDataset
    eval_dataset: InteractionDataset
    test_dataset: InteractionDataset
    w2v: np.ndarray


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
        raise ValueError(f"Embedding dim mismatch: file has {kv.vector_size}, but word_dim={word_dim}")

    w2v = np.random.uniform(-0.05, 0.05, size=(len(vocab), word_dim)).astype(np.float32)
    w2v[0] = 0.0
    for word, idx in vocab.items():
        if word in kv:
            w2v[idx] = kv[word]
    return w2v


def _build_w2v_from_glove_txt(path: str, vocab: Dict[str, int], word_dim: int) -> np.ndarray:
    w2v = np.random.uniform(-0.05, 0.05, size=(len(vocab), word_dim)).astype(np.float32)
    w2v[0] = 0.0
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
    return w2v


def _build_vocab(train_reviews: List[List[List[str]]], max_vocab: int) -> Dict[str, int]:
    counter = Counter()
    for review in train_reviews:
        for sent in review:
            counter.update(sent)

    vocab = {"<pad>": 0, "<unk>": 1}
    for idx, (word, _) in enumerate(counter.most_common(max_vocab), start=2):
        vocab[word] = idx
    return vocab


def _encode_review(
    review_tokens: List[List[str]],
    vocab: Dict[str, int],
    max_sents: int,
    max_sent_len: int,
) -> np.ndarray:
    arr = np.zeros((max_sents, max_sent_len), dtype=np.int64)
    for s_idx, sent in enumerate(review_tokens[:max_sents]):
        ids = [vocab.get(tok, 1) for tok in sent[:max_sent_len]]
        if ids:
            arr[s_idx, : len(ids)] = np.asarray(ids, dtype=np.int64)
    return arr


def _sample_neighbors(rng: np.random.RandomState, pool: List[int], k: int, pad_value: int) -> List[int]:
    if not pool:
        return [pad_value] * k
    if len(pool) >= k:
        idx = rng.choice(len(pool), size=k, replace=False)
        return [pool[i] for i in idx]
    out = pool.copy()
    out.extend([pad_value] * (k - len(out)))
    return out


def _split_arrays(split_df: pd.DataFrame, user2idx: Dict[str, int], item2idx: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = split_df["user_id"].astype(str).map(user2idx).to_numpy(dtype=np.int64)
    i = split_df["item_id"].astype(str).map(item2idx).to_numpy(dtype=np.int64)
    r = split_df["rating"].astype(np.float32).to_numpy()
    return u, i, r


def prepare_data(
    dataset_csv: str,
    train_idx: str,
    eval_idx: str,
    test_idx: str,
    max_sent_len: int = 40,
    max_sents: int = 15,
    max_review_user: int = 40,
    max_review_item: int = 50,
    max_neighbor: int = 75,
    max_vocab: int = 50000,
    word_dim: int = 300,
    pretrained_emb_path: str = "",
    pretrained_emb_format: str = "auto",
    seed: int = 42,
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

    # Keep the same id universe as the experiment splits only.
    all_df = pd.concat([train_df, eval_df, test_df], ignore_index=True)
    user_values = all_df["user_id"].astype(str).drop_duplicates().tolist()
    item_values = all_df["item_id"].astype(str).drop_duplicates().tolist()
    user2idx = {u: i for i, u in enumerate(user_values)}
    item2idx = {it: i for i, it in enumerate(item_values)}

    train_reviews_tokens: List[List[List[str]]] = []
    user_review_mats = defaultdict(list)
    item_review_mats = defaultdict(list)
    user_to_items = defaultdict(list)
    item_to_users = defaultdict(list)

    for row in train_df.itertuples(index=False):
        u = user2idx[str(getattr(row, "user_id"))]
        i = item2idx[str(getattr(row, "item_id"))]
        review_tokens = _split_review(getattr(row, review_col))
        train_reviews_tokens.append(review_tokens)
        user_review_mats[u].append(review_tokens)
        item_review_mats[i].append(review_tokens)
        user_to_items[u].append(i)
        item_to_users[i].append(u)

    vocab = _build_vocab(train_reviews_tokens, max_vocab=max_vocab)
    vocab_size = len(vocab)

    user_docs = np.zeros((len(user_values), max_review_user, max_sents, max_sent_len), dtype=np.int64)
    item_docs = np.zeros((len(item_values), max_review_item, max_sents, max_sent_len), dtype=np.int64)

    unk_review = _encode_review([["<unk>"]], vocab, max_sents=max_sents, max_sent_len=max_sent_len)

    for u in range(len(user_values)):
        reviews = user_review_mats[u][:max_review_user]
        if not reviews:
            user_docs[u, 0] = unk_review
            continue
        for r_idx, review_tokens in enumerate(reviews):
            user_docs[u, r_idx] = _encode_review(review_tokens, vocab, max_sents=max_sents, max_sent_len=max_sent_len)

    for i in range(len(item_values)):
        reviews = item_review_mats[i][:max_review_item]
        if not reviews:
            item_docs[i, 0] = unk_review
            continue
        for r_idx, review_tokens in enumerate(reviews):
            item_docs[i, r_idx] = _encode_review(review_tokens, vocab, max_sents=max_sents, max_sent_len=max_sent_len)

    pad_user = len(user_values)
    pad_item = len(item_values)

    rng = np.random.RandomState(seed)

    user_neighbors = np.full((len(user_values), max_neighbor), pad_item, dtype=np.int64)
    item_neighbors = np.full((len(item_values), max_neighbor), pad_user, dtype=np.int64)
    user_item_user = np.full((len(user_values), max_neighbor, max_neighbor), pad_user, dtype=np.int64)
    item_user_item = np.full((len(item_values), max_neighbor, max_neighbor), pad_item, dtype=np.int64)

    for u in range(len(user_values)):
        uniq_items = list(dict.fromkeys(user_to_items[u]))
        sampled_items = _sample_neighbors(rng, uniq_items, max_neighbor, pad_item)
        user_neighbors[u] = np.asarray(sampled_items, dtype=np.int64)

        for n_idx, it in enumerate(sampled_items):
            if it == pad_item:
                continue
            neigh_users = list(dict.fromkeys(item_to_users[it]))
            sampled_users = _sample_neighbors(rng, neigh_users, max_neighbor, pad_user)
            user_item_user[u, n_idx] = np.asarray(sampled_users, dtype=np.int64)

    for i in range(len(item_values)):
        uniq_users = list(dict.fromkeys(item_to_users[i]))
        sampled_users = _sample_neighbors(rng, uniq_users, max_neighbor, pad_user)
        item_neighbors[i] = np.asarray(sampled_users, dtype=np.int64)

        for n_idx, u in enumerate(sampled_users):
            if u == pad_user:
                continue
            neigh_items = list(dict.fromkeys(user_to_items[u]))
            sampled_items = _sample_neighbors(rng, neigh_items, max_neighbor, pad_item)
            item_user_item[i, n_idx] = np.asarray(sampled_items, dtype=np.int64)

    tr_u, tr_i, tr_r = _split_arrays(train_df, user2idx, item2idx)
    va_u, va_i, va_r = _split_arrays(eval_df, user2idx, item2idx)
    te_u, te_i, te_r = _split_arrays(test_df, user2idx, item2idx)

    if pretrained_emb_path:
        if not os.path.exists(pretrained_emb_path):
            raise FileNotFoundError(f"Embedding path not found: {pretrained_emb_path}")
        emb_format = _infer_emb_format(pretrained_emb_path) if pretrained_emb_format == "auto" else pretrained_emb_format
        if emb_format == "npy":
            w2v = np.load(pretrained_emb_path).astype(np.float32)
            if w2v.shape[1] != word_dim:
                raise ValueError(f"Embedding dim mismatch: file has {w2v.shape[1]}, but word_dim={word_dim}")
            if w2v.shape[0] < vocab_size:
                pad = np.random.uniform(-0.05, 0.05, size=(vocab_size - w2v.shape[0], word_dim)).astype(np.float32)
                pad[0] = 0.0
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
        w2v = np.random.uniform(-0.05, 0.05, size=(vocab_size, word_dim)).astype(np.float32)
        w2v[0] = 0.0

    return PreparedData(
        user_num=len(user_values),
        item_num=len(item_values),
        vocab_size=vocab_size,
        user_docs=torch.from_numpy(user_docs),
        item_docs=torch.from_numpy(item_docs),
        user_neighbors=torch.from_numpy(user_neighbors),
        item_neighbors=torch.from_numpy(item_neighbors),
        user_item_user=torch.from_numpy(user_item_user),
        item_user_item=torch.from_numpy(item_user_item),
        train_dataset=InteractionDataset(tr_u, tr_i, tr_r),
        eval_dataset=InteractionDataset(va_u, va_i, va_r),
        test_dataset=InteractionDataset(te_u, te_i, te_r),
        w2v=w2v,
    )
