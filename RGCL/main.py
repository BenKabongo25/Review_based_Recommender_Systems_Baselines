import argparse
import json
import numpy as np
import os
import random
import string
import torch
import torch.nn as nn

from data import RGCLDataset
from model import RGCL


def evaluate(params, model, dataset, segment="valid"):
    possible_rating_values = dataset.possible_rating_values

    if segment == "valid":
        rating_values = dataset.valid_truths
        enc_graph = dataset.valid_enc_graph
        dec_graph = dataset.valid_dec_graph
    elif segment == "test":
        rating_values = dataset.test_truths
        enc_graph = dataset.test_enc_graph
        dec_graph = dataset.test_dec_graph
    else:
        raise NotImplementedError

    nd_possible_rating_values = torch.FloatTensor(possible_rating_values).to(params.device)

    model.eval()
    with torch.no_grad():
        pred_ratings, _, _, _ = model(
            enc_graph,
            dec_graph,
            dataset.user_feature,
            dataset.item_feature,
        )
        mse = ((pred_ratings - rating_values) ** 2.0).mean().item()
        rmse = np.sqrt(mse)
        mae = (pred_ratings - rating_values).abs().mean().item()
    return rmse, mse, mae


def main(params):
    print(params)
    dataset = RGCLDataset(
        dataset_csv=params.dataset_csv,
        train_idx_path=params.train_idx,
        eval_idx_path=params.eval_idx,
        test_idx_path=params.test_idx,
        review_feat_path=params.review_feat_path,
        device=params.device,
        review_fea_size=params.review_feat_size,
        symm=True,
    )
    print("Loading data finished ...\n")

    params.src_in_units = dataset.user_feature_shape[1]
    params.dst_in_units = dataset.item_feature_shape[1]
    params.rating_vals = dataset.possible_rating_values

    model = RGCL(params).to(params.device)
    nd_possible_rating_values = torch.FloatTensor(dataset.possible_rating_values).to(params.device)

    rating_loss_net = nn.MSELoss()
    learning_rate = params.train_lr
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print("Loading network finished ...\n")

    train_gt_labels = dataset.train_truths.float()
    train_gt_ratings = dataset.train_truths.float()

    best_valid_rmse = np.inf
    no_better_valid = 0
    best_test = None

    dataset.train_enc_graph = dataset.train_enc_graph.int().to(params.device)
    dataset.train_dec_graph = dataset.train_dec_graph.int().to(params.device)
    dataset.valid_enc_graph = dataset.train_enc_graph
    dataset.valid_dec_graph = dataset.valid_dec_graph.int().to(params.device)
    dataset.test_enc_graph = dataset.test_enc_graph.int().to(params.device)
    dataset.test_dec_graph = dataset.test_dec_graph.int().to(params.device)

    print("Start training ...")
    for iter_idx in range(1, params.train_max_iter + 1):
        model.train()
        optimizer.zero_grad()

        pred_ratings1, ed_mi1, user1, item1 = model(
            dataset.train_enc_graph,
            dataset.train_dec_graph,
            dataset.user_feature,
            dataset.item_feature,
        )
        pred_ratings2, ed_mi2, user2, item2 = model(
            dataset.train_enc_graph,
            dataset.train_dec_graph,
            dataset.user_feature,
            dataset.item_feature,
        )
        loss1 = rating_loss_net(pred_ratings1, train_gt_labels).mean()
        loss2 = rating_loss_net(pred_ratings2, train_gt_labels).mean()
        user_mi_loss = model.contrast_loss(user1, user2).mean()
        item_mi_loss = model.contrast_loss(item1, item2).mean()
        r_loss = (loss1 + loss2) / 2

        nd_loss = (user_mi_loss + item_mi_loss) / 2
        ed_loss = (ed_mi1.mean() + ed_mi2.mean()) / 2

        total_loss = r_loss + params.nd_alpha * nd_loss + params.ed_alpha * ed_loss
        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), params.train_grad_clip)
        optimizer.step()

        real_pred_ratings = pred_ratings1

        train_rmse = ((real_pred_ratings - train_gt_ratings) ** 2).mean().sqrt()

        valid_rmse, _, _ = evaluate(params=params, model=model, dataset=dataset, segment="valid")
        logging_str = (
            f"Iter={iter_idx:>4d}, Train_RMSE={train_rmse:.4f}, ED_MI={ed_loss:.4f}, "
            f"ND_MI={nd_loss:.4f}, Valid_RMSE={valid_rmse:.4f}, "
        )

        if valid_rmse < best_valid_rmse:
            best_valid_rmse = valid_rmse
            no_better_valid = 0

            test_rmse, test_mse, test_mae = evaluate(
                params=params, model=model, dataset=dataset, segment="test"
            )
            best_test = {"rmse": test_rmse, "mse": test_mse, "mae": test_mae}
            print(f"Test RMSE: {test_rmse:.4f}, MSE: {test_mse:.4f}, MAE: {test_mae:.4f}")

            with open(
                os.path.join(params.output_dir, f"{params.dataset_name}_RGCL_results_{params.seed}.json"),
                "w",
            ) as f:
                json.dump(best_test, f, indent=4)

            logging_str += f"Test_RMSE={test_rmse:.4f}"
        else:
            no_better_valid += 1
            if (
                no_better_valid > params.train_early_stopping_patience
                and learning_rate <= params.train_min_lr
            ):
                print("Early stopping threshold reached. Stop training.")
                break
            if no_better_valid > params.train_decay_patience:
                new_lr = max(learning_rate * params.train_lr_decay_factor, params.train_min_lr)
                if new_lr < learning_rate:
                    learning_rate = new_lr
                    print(f"\tChange the LR to {new_lr:g}")
                    for p in optimizer.param_groups:
                        p["lr"] = learning_rate
                    no_better_valid = 0

        print(logging_str)

    if best_test is not None:
        print(
            f"Best Test Metrics -> RMSE: {best_test['rmse']:.4f}, "
            f"MSE: {best_test['mse']:.4f}, MAE: {best_test['mae']:.4f}"
        )


def parse_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    try:
        device_idx = int(device_arg)
        if device_idx == -1:
            return torch.device("cpu")
        if device_idx >= 0:
            return torch.device(f"cuda:{device_idx}")
    except ValueError:
        pass
    return torch.device(device_arg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RGCL")
    parser.add_argument("--device", type=str, default="0", help="Device: '-1'/'cpu' for CPU, '0' or 'cuda:0' for GPU.")
    parser.add_argument("--model_save_path", type=str, help="The model saving path")

    parser.add_argument("--dataset_name", type=str, required=True, help="dataset name")
    parser.add_argument("--dataset_csv", required=True, help="Path to CSV dataset.")
    parser.add_argument("--train_idx", required=True, help="Path to train indices .npy.")
    parser.add_argument("--eval_idx", required=True, help="Path to eval/valid indices .npy.")
    parser.add_argument("--test_idx", required=True, help="Path to test indices .npy.")
    parser.add_argument("--review_feat_path", required=True, help="Path to review feature .pkl file.")
    parser.add_argument("--review_feat_size", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default="log", help="Directory to save results.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--gcn_dropout", type=float, default=0.7)
    parser.add_argument("--train_max_iter", type=int, default=500)
    parser.add_argument("--train_grad_clip", type=float, default=1.0)
    parser.add_argument("--train_lr", type=float, default=0.01)
    parser.add_argument("--train_min_lr", type=float, default=0.001)
    parser.add_argument("--train_lr_decay_factor", type=float, default=0.5)
    parser.add_argument("--train_decay_patience", type=int, default=20)
    parser.add_argument("--train_early_stopping_patience", type=int, default=50)
    parser.add_argument("--share_param", default=False, action="store_true")

    parser.add_argument("--ed_alpha", type=float, default=1.0)
    parser.add_argument("--nd_alpha", type=float, default=0.3)

    args = parser.parse_args()
    args.model_short_name = "RGCL"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)

    args.device = parse_device(args.device)

    if args.model_save_path is None:
        args.model_save_path = os.path.join(args.output_dir, f"{args.dataset_name}_{args.model_short_name}_{args.seed}.pth")

    args.gcn_agg_units = args.review_feat_size
    args.gcn_out_units = args.review_feat_size

    main(args)
