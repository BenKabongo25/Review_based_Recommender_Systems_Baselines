import argparse
import numpy as np
import os
import pandas as pd
import torch

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertModel, BertTokenizer


class ReviewDataset(Dataset):
    def __init__(self, user, item, rating, review_text, tokenizer):
        self.user = np.array(user).astype(str)
        self.item = np.array(item).astype(str)
        self.r = np.array(rating).astype(np.float32)
        self.tokenizer = tokenizer
        self.docs = review_text

        self.__pre_tokenize()

    def __pre_tokenize(self):
        self.docs = [self.tokenizer.tokenize(x) for x in tqdm(self.docs, desc="pre tokenize")]
        review_length = self.top_review_length(self.docs)
        self.docs = [x[:review_length] for x in self.docs]

    def __getitem__(self, idx):
        return self.user[idx], self.item[idx], self.r[idx], self.docs[idx]

    def __len__(self):
        return len(self.docs)

    @staticmethod
    def top_review_length(docs: list, top=0.8):
        sentence_length = [len(x) for x in docs]
        sentence_length.sort()
        length = sentence_length[int(len(sentence_length) * top)]
        length = 128 if length > 128 else length
        return length


def compute_kernel_bias(vecs, vec_dim):
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    w = np.dot(u, np.diag(1 / np.sqrt(s)))
    return w[:, :vec_dim], -mu


def transform_and_normalize(vecs, kernel=None, bias=None):
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    return vecs / (vecs**2).sum(axis=1, keepdims=True) ** 0.5


def load_train_dataframe(dataset_csv: str, train_idx_path: str) -> pd.DataFrame:
    df = pd.read_csv(dataset_csv)
    train_idx = np.load(train_idx_path, allow_pickle=True)
    train_idx = np.asarray(train_idx).reshape(-1)

    if train_idx.dtype.kind not in {"i", "u"}:
        raise ValueError(f"Train indices must be integer dtype, got {train_idx.dtype}.")

    if len(train_idx) == 0:
        raise ValueError("Train indices are empty.")

    if train_idx.min() < 0 or train_idx.max() >= len(df):
        raise IndexError(
            f"Train indices out of bounds for dataset of size {len(df)}."
        )

    return df.iloc[train_idx].reset_index(drop=True)


@torch.no_grad()
def main(params):
    bert_tokenizer = BertTokenizer.from_pretrained(
        params.pretrained_weight_shortcut,
        model_max_length=params.review_max_length,
    )

    train_data = load_train_dataframe(params.dataset_csv, params.train_idx)

    train_dataset = ReviewDataset(
        train_data["user_id"].tolist(),
        train_data["item_id"].tolist(),
        train_data["rating"].tolist(),
        train_data["review"].astype(str).tolist(),
        bert_tokenizer,
    )

    def collate_fn(data):
        u, i, r, tokens = zip(*data)
        tokens = [list(x) for x in tokens]
        encoding = bert_tokenizer(
            tokens,
            return_tensors="pt",
            padding=True,
            truncation=True,
            is_split_into_words=True,
        )

        return (
            u,
            i,
            r,
            encoding["input_ids"],
            encoding["attention_mask"],
        )

    data_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn)

    bert = BertModel.from_pretrained(params.pretrained_weight_shortcut).to(params.device)
    bert.config.output_hidden_states = True

    vecs = []
    for _, _, _, input_ids, mask in tqdm(data_loader):
        input_ids = input_ids.to(params.device)
        mask = mask.to(params.device)

        outputs = bert(input_ids, mask)
        output1 = outputs[2][-2]
        output2 = outputs[2][-1]
        last2 = output1 + output2 / 2
        last2 = torch.sum(mask.unsqueeze(-1) * last2, dim=1) / mask.sum(dim=1, keepdims=True)
        vecs.append(last2.cpu().numpy())

    vecs = np.vstack(vecs)
    kernel, bias = compute_kernel_bias(vecs, params.vec_dim)
    vecs = transform_and_normalize(vecs, kernel, bias)
    vecs = torch.from_numpy(vecs)

    ui = list(zip(train_data["user_id"].tolist(), train_data["item_id"].tolist()))
    vecs = dict(zip(ui, vecs))

    os.makedirs(os.path.dirname(params.feat_save_path), exist_ok=True)
    torch.save(vecs, params.feat_save_path)
    print(f"Saved embeddings to {params.feat_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_csv", required=True, help="Path to CSV dataset.")
    parser.add_argument("--train_idx", required=True, help="Path to train indices .npy.")
    parser.add_argument("--output_dir", required=True, help="Output directory for embeddings.")
    parser.add_argument("--pretrained_weight_shortcut", type=str, default="bert-base-uncased")
    parser.add_argument("--model_short_name", type=str, default="BERT-Whitening")
    parser.add_argument("--vec_dim", type=int, default=64)
    parser.add_argument("--review_max_length", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    args.feat_save_path = (
        f"{args.output_dir}/review_embeddings_{args.vec_dim}.pkl"
    )
    main(args)
