import argparse
import json
import os
import random
import string
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import RMCLDataset
from model import RMCL


def _load_text_embeddings(emb_path: str) -> Tuple[list, torch.Tensor]:
    obj = torch.load(emb_path, map_location="cpu")
    if not isinstance(obj, dict) or "ids" not in obj or "embeddings" not in obj:
        raise ValueError(f"Invalid embedding file format: {emb_path}")
    ids = [str(x) for x in obj["ids"]]
    emb = obj["embeddings"].float()
    return ids, emb


def _align_embeddings(target_ids: list, loaded_ids: list, loaded_emb: torch.Tensor, name: str) -> torch.Tensor:
    id2idx = {x: i for i, x in enumerate(loaded_ids)}
    missing = [x for x in target_ids if x not in id2idx]
    if missing:
        raise KeyError(f"{name}: missing {len(missing)} ids in text embeddings. Example: {missing[0]}")
    index = torch.tensor([id2idx[x] for x in target_ids], dtype=torch.long)
    return loaded_emb[index]


@torch.no_grad()
def evaluate(
    model: RMCL,
    loader: DataLoader,
    user_text_emb: torch.Tensor,
    item_text_emb: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    preds = []
    trues = []

    for u, i, r in loader:
        u = u.to(device)
        i = i.to(device)
        r = r.to(device)

        user_text = user_text_emb[u]
        item_text = item_text_emb[i]
        pred, _ = model(u, i, user_text, item_text)

        preds.append(pred)
        trues.append(r)

    pred = torch.cat(preds, dim=0)
    true = torch.cat(trues, dim=0)
    mse = torch.mean((pred - true) ** 2).item()
    rmse = float(np.sqrt(mse))
    mae = torch.mean(torch.abs(pred - true)).item()
    return {"rmse": rmse, "mse": mse, "mae": mae}


def train(args):
    print(args)

    dataset = RMCLDataset(
        dataset_csv=args.dataset_csv,
        train_idx=args.train_idx,
        eval_idx=args.eval_idx,
        test_idx=args.test_idx,
        max_doc_words=args.max_doc_words,
    )

    user_ids_loaded, user_emb_loaded = _load_text_embeddings(os.path.join(args.text_emb_dir, "user_text_emb.pt"))
    item_ids_loaded, item_emb_loaded = _load_text_embeddings(os.path.join(args.text_emb_dir, "item_text_emb.pt"))

    user_text_emb = _align_embeddings(dataset.user_ids, user_ids_loaded, user_emb_loaded, "user")
    item_text_emb = _align_embeddings(dataset.item_ids, item_ids_loaded, item_emb_loaded, "item")

    text_dim = int(user_text_emb.shape[1])
    model = RMCL(
        num_users=dataset.num_users,
        num_items=dataset.num_items,
        text_dim=text_dim,
        latent_dim=args.latent_dim,
        num_intentions=args.num_intentions,
        dropout=args.dropout,
        lambda_cl=args.lambda_cl,
        eta_sim=args.eta_sim,
        mu_ind=args.mu_ind,
    ).to(args.device)

    user_text_emb = user_text_emb.to(args.device)
    item_text_emb = item_text_emb.to(args.device)

    train_loader = DataLoader(dataset.get_interaction_dataset("train"), batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(dataset.get_interaction_dataset("eval"), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(dataset.get_interaction_dataset("test"), batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_eval_rmse = float("inf")
    best_metrics = None
    bad_count = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = {"loss": 0.0, "rate": 0.0, "cl": 0.0, "sim": 0.0, "ind": 0.0}
        n_batch = 0

        for u, i, r in train_loader:
            u = u.to(args.device)
            i = i.to(args.device)
            r = r.to(args.device)

            user_text = user_text_emb[u]
            item_text = item_text_emb[i]

            pred, aux = model(u, i, user_text, item_text)
            losses = model.compute_loss(pred, r, aux)

            optimizer.zero_grad()
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            n_batch += 1
            for k in running:
                running[k] += float(losses[k].detach().item())

        for k in running:
            running[k] /= max(n_batch, 1)

        eval_metrics = evaluate(model, eval_loader, user_text_emb, item_text_emb, args.device)

        log = (
            f"Epoch={epoch:03d} "
            f"TrainLoss={running['loss']:.4f} Rate={running['rate']:.4f} "
            f"CL={running['cl']:.4f} SIM={running['sim']:.4f} IND={running['ind']:.4f} "
            f"EvalRMSE={eval_metrics['rmse']:.4f}"
        )

        if eval_metrics["rmse"] < best_eval_rmse:
            best_eval_rmse = eval_metrics["rmse"]
            bad_count = 0
            test_metrics = evaluate(model, test_loader, user_text_emb, item_text_emb, args.device)
            best_metrics = {
                "eval": eval_metrics,
                "test": test_metrics,
                "epoch": epoch,
            }

            if args.model_save_path:
                torch.save(model.state_dict(), args.model_save_path)

            with open(os.path.join(args.output_dir, f"{args.dataset_name}_RMCL_results_{args.seed}.json"), "w") as f:
                json.dump(best_metrics, f, indent=2)

            log += f" TestRMSE={test_metrics['rmse']:.4f}"
        else:
            bad_count += 1
            if bad_count >= args.early_stop_patience:
                print("Early stopping triggered.")
                print(log)
                break

        print(log)

    if best_metrics is not None:
        print(
            f"Best Epoch={best_metrics['epoch']} "
            f"EvalRMSE={best_metrics['eval']['rmse']:.4f} "
            f"TestRMSE={best_metrics['test']['rmse']:.4f}"
        )


def main():
    parser = argparse.ArgumentParser(description="RMCL training")
    parser.add_argument("-dn", "--dataset_name", type=str, required=True)
    parser.add_argument("--dataset_csv", required=True, help="Path to CSV dataset.")
    parser.add_argument("--train_idx", required=True, help="Path to train indices .npy.")
    parser.add_argument("--eval_idx", required=True, help="Path to eval indices .npy.")
    parser.add_argument("--test_idx", required=True, help="Path to test indices .npy.")
    parser.add_argument("--text_emb_dir", required=True, help="Directory containing user_text_emb.pt and item_text_emb.pt")
    parser.add_argument("--output_dir", required=True, help="Directory for results")

    parser.add_argument("--model_save_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    # Parameters based on paper search ranges (Sec. 4.1.3).
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num_intentions", type=int, default=20)

    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--early_stop_patience", type=int, default=100)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    # Eq. (17) coefficients.
    parser.add_argument("--lambda_cl", type=float, default=1.0)
    parser.add_argument("--eta_sim", type=float, default=1.0)
    parser.add_argument("--mu_ind", type=float, default=1.0)

    parser.add_argument("--max_doc_words", type=int, default=300)

    args = parser.parse_args()
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    if args.model_save_path is None:
        args.model_save_path = os.path.join(
            args.output_dir,
            f"{args.dataset_name}_RMCL_{args.seed}_{''.join(random.choices(string.ascii_uppercase + string.digits, k=2))}.pth",
        )

    train(args)


if __name__ == "__main__":
    main()
