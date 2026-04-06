import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import prepare_data
from model import NARRE


def now() -> str:
    return str(time.strftime("%Y-%m-%d %H:%M:%S"))


def build_batch(pdata, uids: torch.Tensor, iids: torch.Tensor, device: torch.device):
    user_reviews = pdata.users_review_list[uids].to(device)
    item_reviews = pdata.items_review_list[iids].to(device)
    user_item_ids = pdata.user2itemid_list[uids].to(device)
    item_user_ids = pdata.item2userid_list[iids].to(device)
    return (
        user_reviews,
        item_reviews,
        uids.to(device),
        iids.to(device),
        user_item_ids,
        item_user_ids,
    )


@torch.no_grad()
def evaluate(model: NARRE, loader: DataLoader, pdata, device: torch.device):
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    total_n = 0

    for uids, iids, scores in loader:
        scores = scores.to(device)
        batch = build_batch(pdata, uids, iids, device)
        preds = model(*batch)

        total_mse += torch.sum((preds - scores) ** 2).item()
        total_mae += torch.sum(torch.abs(preds - scores)).item()
        total_n += scores.size(0)

    mse = total_mse / max(1, total_n)
    mae = total_mae / max(1, total_n)
    rmse = math.sqrt(mse)
    model.train()
    return {"mse": mse, "rmse": rmse, "mae": mae}


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    pdata = prepare_data(
        dataset_csv=args.dataset_csv,
        train_idx=args.train_idx,
        eval_idx=args.eval_idx,
        test_idx=args.test_idx,
        max_vocab=args.max_vocab,
        p_review=args.p_review,
        word_dim=args.word_dim,
        pretrained_emb_path=args.pretrained_emb_path,
        pretrained_emb_format=args.pretrained_emb_format,
    )

    args.user_num = pdata.user_num
    args.item_num = pdata.item_num
    args.vocab_size = pdata.vocab_size
    args.w2v_matrix = pdata.w2v

    model = NARRE(args).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse_func = nn.MSELoss()

    train_loader = DataLoader(pdata.train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(pdata.eval_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(pdata.test_dataset, batch_size=args.batch_size, shuffle=False)

    best_eval_mse = float("inf")
    best_result = None
    no_improve = 0

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        total_train_loss = 0.0
        total_train_n = 0

        for uids, iids, scores in train_loader:
            scores = scores.to(args.device)
            batch = build_batch(pdata, uids, iids, args.device)

            optimizer.zero_grad()
            preds = model(*batch)
            loss = mse_func(preds, scores)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * scores.size(0)
            total_train_n += scores.size(0)

        train_mse = total_train_loss / max(1, total_train_n)
        eval_metrics = evaluate(model, eval_loader, pdata, args.device)

        msg = (
            f"{now()} Epoch={epoch:03d} "
            f"TrainMSE={train_mse:.4f} EvalMSE={eval_metrics['mse']:.4f} "
            f"EvalRMSE={eval_metrics['rmse']:.4f} EvalMAE={eval_metrics['mae']:.4f}"
        )

        if eval_metrics["mse"] < best_eval_mse:
            best_eval_mse = eval_metrics["mse"]
            no_improve = 0
            test_metrics = evaluate(model, test_loader, pdata, args.device)
            best_result = {"epoch": epoch, "eval": eval_metrics, "test": test_metrics}
            torch.save(model.state_dict(), args.model_save_path)
            with open(args.result_json_path, "w") as f:
                json.dump(best_result, f, indent=2)
            msg += f" TestRMSE={test_metrics['rmse']:.4f}"
        else:
            no_improve += 1

        print(msg)
        if no_improve >= args.early_stopping_patience:
            print(f"Early stopping at epoch {epoch} (patience={args.early_stopping_patience}).")
            break

    if best_result is not None:
        print(
            f"Best epoch={best_result['epoch']} "
            f"EvalRMSE={best_result['eval']['rmse']:.4f} "
            f"TestRMSE={best_result['test']['rmse']:.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NARRE training")

    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--dataset_csv", required=True)
    parser.add_argument("--train_idx", required=True)
    parser.add_argument("--eval_idx", required=True)
    parser.add_argument("--test_idx", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--early_stopping_patience", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)

    parser.add_argument("--word_dim", type=int, default=300)
    parser.add_argument("--max_vocab", type=int, default=30000)
    parser.add_argument("--p_review", type=float, default=0.9)

    parser.add_argument("--filters_num", type=int, default=100)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.5)

    parser.add_argument("--pretrained_emb_path", type=str, default="")
    parser.add_argument(
        "--pretrained_emb_format",
        type=str,
        default="auto",
        choices=["auto", "word2vec_bin", "glove_txt", "npy"],
    )

    parser.add_argument("--model_name", type=str, default="NARRE")

    args = parser.parse_args()
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)
    args.model_save_path = os.path.join(args.output_dir, f"{args.dataset_name}_{args.model_name}_{args.seed}.pth")
    args.result_json_path = os.path.join(
        args.output_dir,
        f"{args.dataset_name}_{args.model_name}_results_{args.seed}.json",
    )

    main(args)
