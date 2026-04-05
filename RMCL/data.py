import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


WORD_RE = re.compile(r"[A-Za-z]+")

# Lightweight stopword set to mirror the paper's stopword removal step.
# (SIGIR'23 RMCL, Sec. 4.1.1)
EN_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he",
    "in", "is", "it", "its", "of", "on", "that", "the", "to", "was", "were", "will",
    "with", "this", "these", "those", "or", "if", "then", "than", "but", "so", "we",
    "you", "i", "they", "them", "our", "your", "their", "me", "my", "mine", "his",
    "her", "hers", "who", "whom", "which", "what", "when", "where", "why", "how",
    "do", "does", "did", "done", "have", "had", "having", "can", "could", "should",
    "would", "may", "might", "must", "not", "no", "yes", "up", "down", "out", "over",
    "under", "again", "further", "once", "very", "just", "into", "about", "after", "before",
}


@dataclass
class InteractionSplit:
    user_idx: torch.LongTensor
    item_idx: torch.LongTensor
    ratings: torch.FloatTensor


class InteractionDataset(Dataset):
    def __init__(self, split: InteractionSplit):
        self.user_idx = split.user_idx
        self.item_idx = split.item_idx
        self.ratings = split.ratings

    def __len__(self):
        return self.ratings.shape[0]

    def __getitem__(self, idx):
        return self.user_idx[idx], self.item_idx[idx], self.ratings[idx]


def _load_indices(npy_path: str) -> np.ndarray:
    idx = np.load(npy_path, allow_pickle=True)
    idx = np.asarray(idx).reshape(-1)
    if idx.dtype.kind not in {"i", "u"}:
        raise ValueError(f"Indices must be integer dtype, got {idx.dtype}.")
    if idx.size == 0:
        raise ValueError(f"Empty indices file: {npy_path}")
    return idx


def _normalize_text(text: str, remove_stopwords: bool) -> List[str]:
    tokens = WORD_RE.findall(str(text).lower())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in EN_STOPWORDS]
    return tokens


def _truncate_doc(tokens: List[str], max_words: int) -> str:
    return " ".join(tokens[:max_words])


class RMCLDataset:
    """Dataset helper for RMCL.

    - Loads data from CSV + train/eval/test row indices.
    - Builds user/item id mappings from train+eval+test only.
    - Builds user/item review documents from train interactions.
    """

    def __init__(
        self,
        dataset_csv: str,
        train_idx: str,
        eval_idx: str,
        test_idx: str,
        max_doc_words: int = 300,
        remove_stopwords: bool = True,
    ):
        self.dataset_csv = dataset_csv
        self.max_doc_words = max_doc_words
        self.remove_stopwords = remove_stopwords

        df = pd.read_csv(dataset_csv)
        required = {"user_id", "item_id", "rating"}
        if not required.issubset(df.columns):
            raise ValueError(f"CSV must contain columns {sorted(required)}")

        review_col = "review" if "review" in df.columns else "review_text"
        if review_col not in df.columns:
            raise ValueError("CSV must contain either 'review' or 'review_text' column.")
        self.review_col = review_col

        train_ids = _load_indices(train_idx)
        eval_ids = _load_indices(eval_idx)
        test_ids = _load_indices(test_idx)

        for name, ids in (("train", train_ids), ("eval", eval_ids), ("test", test_ids)):
            if ids.min() < 0 or ids.max() >= len(df):
                raise IndexError(f"{name} indices out of bounds for dataset size {len(df)}")

        train_df = df.iloc[train_ids].copy()
        eval_df = df.iloc[eval_ids].copy()
        test_df = df.iloc[test_ids].copy()

        all_df = pd.concat([train_df, eval_df, test_df], axis=0, ignore_index=True)
        self.user_ids: List[str] = all_df["user_id"].astype(str).drop_duplicates().tolist()
        self.item_ids: List[str] = all_df["item_id"].astype(str).drop_duplicates().tolist()
        self.user2idx: Dict[str, int] = {u: i for i, u in enumerate(self.user_ids)}
        self.item2idx: Dict[str, int] = {m: i for i, m in enumerate(self.item_ids)}

        self.num_users = len(self.user_ids)
        self.num_items = len(self.item_ids)

        self.train_split = self._build_split(train_df)
        self.eval_split = self._build_split(eval_df)
        self.test_split = self._build_split(test_df)

        # Documents are built from train interactions to avoid leakage.
        self.user_docs, self.item_docs = self._build_documents(train_df)

    def _build_split(self, split_df: pd.DataFrame) -> InteractionSplit:
        u = split_df["user_id"].astype(str).map(self.user2idx).to_numpy(dtype=np.int64)
        i = split_df["item_id"].astype(str).map(self.item2idx).to_numpy(dtype=np.int64)
        r = split_df["rating"].astype(np.float32).to_numpy()
        return InteractionSplit(
            user_idx=torch.from_numpy(u),
            item_idx=torch.from_numpy(i),
            ratings=torch.from_numpy(r),
        )

    def _build_documents(self, train_df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        user_tokens: Dict[str, List[str]] = {u: [] for u in self.user_ids}
        item_tokens: Dict[str, List[str]] = {m: [] for m in self.item_ids}

        for _, row in train_df.iterrows():
            u = str(row["user_id"])
            m = str(row["item_id"])
            tokens = _normalize_text(str(row[self.review_col]), self.remove_stopwords)
            user_tokens[u].extend(tokens)
            item_tokens[m].extend(tokens)

        user_docs = [_truncate_doc(user_tokens[u], self.max_doc_words) for u in self.user_ids]
        item_docs = [_truncate_doc(item_tokens[m], self.max_doc_words) for m in self.item_ids]
        return user_docs, item_docs

    def get_interaction_dataset(self, split: str) -> InteractionDataset:
        if split == "train":
            return InteractionDataset(self.train_split)
        if split == "eval":
            return InteractionDataset(self.eval_split)
        if split == "test":
            return InteractionDataset(self.test_split)
        raise ValueError(f"Unknown split: {split}")
