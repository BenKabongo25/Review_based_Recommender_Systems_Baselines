import argparse
import json
import os

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

from data import RMCLDataset


class TextDataset(Dataset):
    def __init__(self, ids, docs):
        self.ids = ids
        self.docs = docs

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return self.ids[idx], self.docs[idx]


def encode_docs(ids, docs, tokenizer, encoder, device, batch_size):
    dataset = TextDataset(ids, docs)

    def collate_fn(batch):
        batch_ids, batch_docs = zip(*batch)
        enc = tokenizer(
            list(batch_docs),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        return list(batch_ids), enc["input_ids"], enc["attention_mask"]

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    all_ids = []
    all_vecs = []
    encoder.eval()
    with torch.no_grad():
        for batch_ids, input_ids, attn_mask in tqdm(loader, desc="encode"):
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            out = encoder(input_ids=input_ids, attention_mask=attn_mask)
            cls = out.last_hidden_state[:, 0, :]
            all_ids.extend(batch_ids)
            all_vecs.append(cls.cpu())

    return all_ids, torch.cat(all_vecs, dim=0)


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = RMCLDataset(
        dataset_csv=args.dataset_csv,
        train_idx=args.train_idx,
        eval_idx=args.eval_idx,
        test_idx=args.test_idx,
        max_doc_words=args.max_doc_words,
    )

    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)
    encoder = BertModel.from_pretrained(args.pretrained_model).to(args.device)

    user_ids, user_emb = encode_docs(
        dataset.user_ids,
        dataset.user_docs,
        tokenizer,
        encoder,
        args.device,
        args.batch_size,
    )
    item_ids, item_emb = encode_docs(
        dataset.item_ids,
        dataset.item_docs,
        tokenizer,
        encoder,
        args.device,
        args.batch_size,
    )

    torch.save({"ids": user_ids, "embeddings": user_emb}, os.path.join(args.output_dir, "user_text_emb.pt"))
    torch.save({"ids": item_ids, "embeddings": item_emb}, os.path.join(args.output_dir, "item_text_emb.pt"))

    with open(os.path.join(args.output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "pretrained_model": args.pretrained_model,
                "max_doc_words": args.max_doc_words,
                "text_dim": int(user_emb.shape[1]),
            },
            f,
            indent=2,
        )

    print(f"Saved user/item text embeddings to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute RMCL BERT text embeddings.")
    parser.add_argument("--dataset_csv", required=True, help="Path to CSV dataset.")
    parser.add_argument("--train_idx", required=True, help="Path to train indices .npy.")
    parser.add_argument("--eval_idx", required=True, help="Path to eval indices .npy.")
    parser.add_argument("--test_idx", required=True, help="Path to test indices .npy.")
    parser.add_argument("--output_dir", required=True, help="Output directory for text embeddings.")
    parser.add_argument("--pretrained_model", default="bert-base-uncased")
    parser.add_argument("--max_doc_words", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    main(args)
