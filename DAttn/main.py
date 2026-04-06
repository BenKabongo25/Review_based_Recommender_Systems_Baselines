import argparse
import json
import math
import os
import random
import string
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import prepare_data
from model import RecommenderModel


def now():
    return str(time.strftime('%Y-%m-%d %H:%M:%S'))


def parse_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    try:
        did = int(device_arg)
        if did == -1:
            return torch.device("cpu")
        if did >= 0:
            return torch.device(f"cuda:{did}")
    except ValueError:
        pass
    return torch.device(device_arg)


def unpack_input(opt, pdata, uids, iids):
    user_reviews = pdata.users_review_list[uids]
    user_item2id = pdata.user2itemid_list[uids]
    user_doc = pdata.user_doc[uids]

    item_reviews = pdata.items_review_list[iids]
    item_user2id = pdata.item2userid_list[iids]
    item_doc = pdata.item_doc[iids]

    data = [
        user_reviews.to(opt.device),
        item_reviews.to(opt.device),
        uids.to(opt.device),
        iids.to(opt.device),
        user_item2id.to(opt.device),
        item_user2id.to(opt.device),
        user_doc.to(opt.device),
        item_doc.to(opt.device),
    ]
    return data


@torch.no_grad()
def evaluate(model, loader, pdata, opt):
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    n = 0
    for uids, iids, scores in loader:
        scores = scores.to(opt.device)
        datas = unpack_input(opt, pdata, uids, iids)
        output = model(datas)

        total_mse += torch.sum((output - scores) ** 2).item()
        total_mae += torch.sum(torch.abs(output - scores)).item()
        n += scores.shape[0]

    mse = total_mse / max(n, 1)
    rmse = math.sqrt(mse)
    mae = total_mae / max(n, 1)
    model.train()
    return {"mse": mse, "rmse": rmse, "mae": mae}


def train(opt):
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opt.seed)

    pdata = prepare_data(
        dataset_csv=opt.dataset_csv,
        train_idx=opt.train_idx,
        eval_idx=opt.eval_idx,
        test_idx=opt.test_idx,
        doc_len=opt.doc_len,
        max_vocab=opt.max_vocab,
        p_review=opt.p_review,
        word_dim=opt.word_dim,
        w2v_path=opt.w2v_path,
    )

    opt.user_num = pdata.user_num
    opt.item_num = pdata.item_num
    opt.vocab_size = pdata.vocab_size
    opt.w2v_matrix = pdata.w2v if opt.use_word_embedding else None

    model = RecommenderModel(opt).to(opt.device)

    train_loader = DataLoader(pdata.train_dataset, batch_size=opt.batch_size, shuffle=True)
    eval_loader = DataLoader(pdata.eval_dataset, batch_size=opt.batch_size, shuffle=False)
    test_loader = DataLoader(pdata.test_dataset, batch_size=opt.batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    mse_func = nn.MSELoss()
    mae_func = nn.L1Loss()
    smooth_mae_func = nn.SmoothL1Loss()

    best_eval_rmse = float("inf")
    best_metrics = None
    bad_count = 0

    for epoch in range(1, opt.num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0

        for uids, iids, scores in train_loader:
            scores = scores.to(opt.device)
            datas = unpack_input(opt, pdata, uids, iids)

            optimizer.zero_grad()
            output = model(datas)

            mse_loss = mse_func(output, scores)
            mae_loss = mae_func(output, scores)
            smooth_mae_loss = smooth_mae_func(output, scores)

            if opt.loss_method == 'mse':
                loss = mse_loss
            elif opt.loss_method == 'rmse':
                loss = torch.sqrt(mse_loss) / 2.0
            elif opt.loss_method == 'mae':
                loss = mae_loss
            else:
                loss = smooth_mae_loss

            loss.backward()
            optimizer.step()

            total_loss += mse_loss.item() * scores.shape[0]
            total_count += scores.shape[0]

        train_mse = total_loss / max(total_count, 1)
        eval_metrics = evaluate(model, eval_loader, pdata, opt)

        msg = (
            f"{now()} Epoch={epoch:03d} TrainMSE={train_mse:.4f} "
            f"EvalMSE={eval_metrics['mse']:.4f} EvalRMSE={eval_metrics['rmse']:.4f} EvalMAE={eval_metrics['mae']:.4f}"
        )

        if eval_metrics["rmse"] < best_eval_rmse:
            best_eval_rmse = eval_metrics["rmse"]
            bad_count = 0
            test_metrics = evaluate(model, test_loader, pdata, opt)
            best_metrics = {
                "epoch": epoch,
                "eval": eval_metrics,
                "test": test_metrics,
            }
            torch.save(model.state_dict(), opt.model_save_path)
            with open(os.path.join(opt.output_dir, f"{opt.dataset_name}_{opt.model_name}_results_{opt.seed}.json"), "w") as f:
                json.dump(best_metrics, f, indent=2)
            msg += f" TestRMSE={test_metrics['rmse']:.4f}"
        else:
            bad_count += 1
            if bad_count >= opt.early_stop_patience:
                print(msg)
                print("Early stopping.")
                break

        print(msg)

    if best_metrics is not None:
        print(
            f"Best epoch={best_metrics['epoch']} "
            f"EvalRMSE={best_metrics['eval']['rmse']:.4f} "
            f"TestRMSE={best_metrics['test']['rmse']:.4f}"
        )


def build_parser():
    p = argparse.ArgumentParser(description="Run review-based baseline")
    p.add_argument("--dataset_name", required=True)
    p.add_argument("--dataset_csv", required=True)
    p.add_argument("--train_idx", required=True)
    p.add_argument("--eval_idx", required=True)
    p.add_argument("--test_idx", required=True)
    p.add_argument("--output_dir", required=True)

    p.add_argument("--device", default="0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--loss_method", type=str, default="mse", choices=["mse", "rmse", "mae", "smooth_mae"])
    p.add_argument("--early_stop_patience", type=int, default=8)

    p.add_argument("--word_dim", type=int, default=300)
    p.add_argument("--id_emb_size", type=int, default=32)
    p.add_argument("--query_mlp_size", type=int, default=128)
    p.add_argument("--fc_dim", type=int, default=32)
    p.add_argument("--doc_len", type=int, default=500)
    p.add_argument("--filters_num", type=int, default=100)
    p.add_argument("--kernel_size", type=int, default=3)
    p.add_argument("--drop_out", type=float, default=0.5)

    p.add_argument("--r_id_merge", default="cat", choices=["cat", "sum"])
    p.add_argument("--ui_merge", default="cat", choices=["cat", "add", "dot"])
    p.add_argument("--output", default="lfm", choices=["fm", "lfm", "mlp", "nfm", "sum"])
    p.add_argument("--self_att", action="store_true", default=False)
    p.add_argument("--num_heads", type=int, default=2)

    p.add_argument("--use_word_embedding", action="store_true", default=True)
    p.add_argument("--no_word_embedding", dest="use_word_embedding", action="store_false")
    p.add_argument("--w2v_path", type=str, default="")
    p.add_argument("--max_vocab", type=int, default=50000)
    p.add_argument("--p_review", type=float, default=0.85)

    p.add_argument("--model_name", type=str, default="DAttn")
    p.add_argument("--mpcn_head", type=int, default=3)
    return p


def main():
    parser = build_parser()
    opt = parser.parse_args()
    opt.device = parse_device(opt.device)

    os.makedirs(opt.output_dir, exist_ok=True)
    opt.model_save_path = os.path.join(
        opt.output_dir,
        f"{opt.dataset_name}_{opt.model_name}_{opt.seed}_{''.join(random.choices(string.ascii_uppercase + string.digits, k=2))}.pth",
    )

    train(opt)


if __name__ == "__main__":
    main()
