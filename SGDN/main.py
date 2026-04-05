import argparse
import json
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim

from data import SGDNDataset
from model import SGDN, cal_c_loss


def get_optimizer(opt):
    if opt == 'SGD':
        return optim.SGD
    elif opt == 'Adam':
        return optim.Adam
    elif opt == 'AdamW':
        return optim.AdamW
    else:
        raise NotImplementedError


def evaluate(args, net, dataset, segment="valid"):
    possible_rating_values = dataset.possible_rating_values
    nd_possible_rating_values = torch.FloatTensor(possible_rating_values).to(args.device)

    if segment == "valid":
        rating_values = dataset.valid_truths
        enc_graph = dataset.valid_enc_graphs
        dec_graph = dataset.valid_dec_graph
    elif segment == "test":
        rating_values = dataset.test_truths
        enc_graph = dataset.test_enc_graphs
        dec_graph = dataset.test_dec_graph
    else:
        raise NotImplementedError

    net.eval()
    with torch.no_grad():
        pred_ratings, _, _, _, _, _, _ = net(
            enc_graph,
            dec_graph,
            dataset.review_data_dict,
            save_graph=False,
        )
        mse = ((pred_ratings - rating_values) ** 2.0).mean().item()
        mae = (pred_ratings - rating_values).abs().mean().item()
        rmse = np.sqrt(mse)

    return rmse, mse, mae


def main(params):
    print(params)

    dataset = SGDNDataset(
        dataset_csv=params.dataset_csv,
        train_idx_path=params.train_idx,
        eval_idx_path=params.eval_idx,
        test_idx_path=params.test_idx,
        review_feat_path=params.review_feat_path,
        device=params.device,
        review_fea_size=params.review_feat_size,
        num_factor=params.num_factor,
        symm=params.gcn_agg_norm_symm,
    )
    print("Loading data finished ...\n")

    params.src_in_units = dataset.user_feature_shape[1]
    params.dst_in_units = dataset.item_feature_shape[1]
    params.rating_vals = dataset.possible_rating_values

    net = SGDN(
        params,
        dataset.num_user,
        dataset.num_item,
        dataset.review_data_dict,
        dataset.num_rating,
        dataset.rating_split,
    ).to(params.device)

    nd_possible_rating_values = torch.FloatTensor(dataset.possible_rating_values).to(params.device)
    rating_loss_net = nn.MSELoss()
    learning_rate = params.train_lr
    optimizer = get_optimizer(params.train_optimizer)(net.parameters(), lr=learning_rate)
    print("Loading network finished ...\n")

    train_gt_labels = dataset.train_truths.float()
    train_gt_ratings = dataset.train_truths.float()

    best_valid_rmse = np.inf
    best_test = None
    no_better_valid = 0
    best_iter = -1

    for key in dataset.review_data_dict.keys():
        dataset.review_data_dict[key] = dataset.review_data_dict[key].to(params.device)
    for i, graph in enumerate(dataset.train_enc_graphs):
        dataset.train_enc_graphs[i] = graph.int().to(params.device)
    dataset.train_dec_graph = dataset.train_dec_graph.int().to(params.device)
    for i, graph in enumerate(dataset.valid_enc_graphs):
        dataset.valid_enc_graphs[i] = graph.int().to(params.device)
    dataset.valid_dec_graph = dataset.valid_dec_graph.int().to(params.device)
    for i, graph in enumerate(dataset.test_enc_graphs):
        dataset.test_enc_graphs[i] = graph.int().to(params.device)
    dataset.test_dec_graph = dataset.test_dec_graph.int().to(params.device)
    for i, graph in enumerate(dataset.test_dec_subgraphs):
        dataset.test_dec_subgraphs[i] = graph.int().to(params.device)

    print("Start training ...")
    for iter_idx in range(1, params.train_max_iter + 1):
        net.train()
        pred_ratings1, h_fea1, int_dists1, _, _, _, _ = net(
            dataset.train_enc_graphs,
            dataset.train_dec_graph,
            dataset.review_data_dict,
        )
        pred_ratings2, h_fea2, int_dists2, _, _, _, _ = net(
            dataset.train_enc_graphs,
            dataset.train_dec_graph,
            dataset.review_data_dict,
        )
        loss_cl = cal_c_loss(h_fea1, h_fea2, int_dists1, dataset.rating_split, params.num_pos)
        r_loss = (
            rating_loss_net(pred_ratings1, train_gt_labels).mean()
            + rating_loss_net(pred_ratings2, train_gt_labels).mean()
        ) / 2
        loss = r_loss + loss_cl * params.lamda

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), params.train_grad_clip)
        optimizer.step()

        real_pred_ratings = pred_ratings1
        train_rmse = ((real_pred_ratings - train_gt_ratings) ** 2).mean().sqrt()

        valid_rmse, _, _ = evaluate(args=params, net=net, dataset=dataset, segment="valid")
        logging_str = (
            f"Iter={iter_idx:>4d}, Train_RMSE={train_rmse:.4f}, "
            f"Valid_RMSE={valid_rmse:.4f}, Train_loss={loss:.4f}, "
        )

        if valid_rmse < best_valid_rmse:
            best_valid_rmse = valid_rmse
            no_better_valid = 0
            best_iter = iter_idx

            test_rmse, test_mse, test_mae = evaluate(args=params, net=net, dataset=dataset, segment="test")
            best_test = {"rmse": test_rmse, "mse": test_mse, "mae": test_mae}

            with open(
                os.path.join(params.output_dir, f"{params.dataset_name}_SGDN_results_{params.seed}.json"),
                "w",
            ) as f:
                json.dump(best_test, f, indent=4)

            if params.model_save_path:
                torch.save(net.state_dict(), params.model_save_path)

            logging_str += f"Test_RMSE={test_rmse:.4f}"
        else:
            no_better_valid += 1
            if no_better_valid > params.train_early_stopping_patience and learning_rate <= params.train_min_lr:
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
            f"Best Iter={best_iter}, Best Valid RMSE={best_valid_rmse:.4f}, "
            f"Best Test RMSE={best_test['rmse']:.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SGDN")

    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name")
    parser.add_argument("--dataset_csv", required=True, help="Path to CSV dataset.")
    parser.add_argument("--train_idx", required=True, help="Path to train indices .npy.")
    parser.add_argument("--eval_idx", required=True, help="Path to eval/valid indices .npy.")
    parser.add_argument("--test_idx", required=True, help="Path to test indices .npy.")
    parser.add_argument("--review_feat_path", required=True, help="Path to review feature .pkl file.")
    parser.add_argument("--output_dir", type=str, default="./", help="Directory to save results.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--review_feat_size", type=int, default=64)
    parser.add_argument("--model_activation", type=str, default="leaky")
    parser.add_argument("--num_factor", type=int, default=4)
    parser.add_argument("--num_layer", type=int, default=2)
    parser.add_argument("--num_pos", type=int, default=10)
    parser.add_argument("--lamda", type=float, default=0.1)

    parser.add_argument("--gcn_agg_norm_symm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gcn_agg_accum", type=str, default="sum")
    parser.add_argument("--gcn_dropout", type=float, default=0.8)
    parser.add_argument("--train_max_iter", type=int, default=500)
    parser.add_argument("--train_optimizer", type=str, default="Adam")
    parser.add_argument("--train_grad_clip", type=float, default=1.0)
    parser.add_argument("--train_lr", type=float, default=0.01)
    parser.add_argument("--train_min_lr", type=float, default=0.001)
    parser.add_argument("--train_lr_decay_factor", type=float, default=0.5)
    parser.add_argument("--train_decay_patience", type=int, default=50)
    parser.add_argument("--train_early_stopping_patience", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    args.model_short_name = "SGDN"

    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    args.model_save_path = os.path.join(args.output_dir, f"{args.dataset_name}_{args.model_short_name}_{args.seed}.pth")
    args.gcn_agg_units = args.review_feat_size
    args.gcn_out_units = args.review_feat_size

    main(args)
