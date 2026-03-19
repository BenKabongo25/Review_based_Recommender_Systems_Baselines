import argparse
import copy
import json
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import Model


class ReviewDataset(Dataset):

    def __init__(self, file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        self.data = []
        for user_id, samples in data.items():
            for sample in samples:
                self.data.append((int(user_id), sample[0], sample[1]))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def load_data(train_path, val_path, test_path):
    train_data = ReviewDataset(train_path)
    val_data = ReviewDataset(val_path)
    test_data = ReviewDataset(test_path)
    return train_data, val_data, test_data

def collate_fn(batch):
    user_ids = torch.tensor([x[0] for x in batch], dtype=torch.long)
    ratings = torch.tensor([x[1] for x in batch], dtype=torch.float32)
    items = torch.tensor([x[2] for x in batch], dtype=torch.long)
    return user_ids, items, ratings


def train(model, dataloader, optimizer, criterion, device, epoch):
    model.train()

    total_loss = 0
    total_mse = 0
    total_cl = 0
    total_aa = 0

    for user_ids, item_ids, ratings in tqdm(dataloader):
        user_ids, item_ids, ratings = user_ids.to(device), item_ids.to(device), ratings.to(device)
        
        optimizer.zero_grad()
        outputs, reg_loss, cl_loss, aa_loss = model(user_ids, item_ids, ratings, clip=0)

        loss_ = criterion(outputs, ratings)
        loss = loss_ + args.reg_lr*reg_loss + args.cl_lr*cl_loss + args.aa_lr*aa_loss
        loss.backward() 
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        total_loss += loss.item()
        total_mse += loss_.item()
        total_cl += args.cl_lr*cl_loss
        total_aa += args.aa_lr*aa_loss

    return total_loss / len(dataloader), total_mse / len(dataloader), total_cl / len(dataloader), total_aa / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for user_ids, item_ids, ratings in dataloader:
            user_ids, item_ids, ratings = user_ids.to(device), item_ids.to(device), ratings.to(device)
            outputs, _, _, _ = model(user_ids, item_ids, ratings)
            loss = criterion(outputs, ratings)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate_rmse(model, dataloader, device):
    model.eval()

    predictions = []
    actuals = []

    total_mae = 0
    total_mse = 0
    total_rmse = 0

    with torch.no_grad():
        for user_ids, item_ids, ratings in dataloader:
            user_ids, item_ids, ratings = user_ids.to(device), item_ids.to(device), ratings.to(device)
            
            outputs, _, _, _ = model(user_ids, item_ids, ratings, clip=1)

            mae = (ratings - outputs).abs().mean()
            mse = nn.MSELoss()(outputs, ratings)
            rmse = np.sqrt(mse.item())

            total_mae = total_mae + mae.item()
            total_mse = total_mse + mse.item()
            total_rmse = total_rmse + rmse

            predictions.extend(outputs.cpu().numpy())
            actuals.extend(ratings.cpu().numpy())

    return total_rmse / len(dataloader), total_mse / len(dataloader), total_mae / len(dataloader)

def save_model(model_or_state_dict, path):
    if isinstance(model_or_state_dict, nn.Module):
        state_dict = model_or_state_dict.state_dict()
    elif isinstance(model_or_state_dict, dict):
        state_dict = model_or_state_dict
    else:
        raise TypeError("save_model expects an nn.Module or a state_dict-like dict.")
    torch.save(state_dict, path)

def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)

def load_ratings(dataset, max_reviewer_index, max_asin_index):
    with open(dataset, 'r') as file:
        data = json.load(file)

    reviewer_ratings = np.zeros((max_reviewer_index, max_asin_index))
    asin_ratings = np.zeros((max_asin_index, max_reviewer_index))

    for reviewer_index, ratings in data.items():
        reviewer_index = int(reviewer_index)
        for rating, asin_index in ratings:
            reviewer_ratings[reviewer_index, asin_index] = rating
            asin_ratings[asin_index, reviewer_index] = rating

    return reviewer_ratings.astype('int'), asin_ratings.astype('int')

def main():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    dataset = args.dataset
    print(f'Dataset: {dataset}')
    os.makedirs(args.output_dir, exist_ok=True)

    train_path = os.path.join(args.dataset_dir, 'train.json')
    val_path = os.path.join(args.dataset_dir, 'eval.json')
    test_path = os.path.join(args.dataset_dir, 'test.json')
    
    train_data, val_data, test_data = load_data(train_path, val_path, test_path)

    user2id = json.load(open(os.path.join(args.dataset_dir, 'user2id.json'), 'r'))
    item2id = json.load(open(os.path.join(args.dataset_dir, 'item2id.json'), 'r'))
    num_users = len(user2id)
    num_items = len(item2id)

    #num_users = max([int(user_id) for user_id, _, _ in train_data.data]) + 1
    #num_items = max([asin for _, _, asin in train_data.data]) + 1
    print(f'Users: {num_users}')
    print(f'Items: {num_items}')
    
    train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=args.batch, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=args.batch, shuffle=False, collate_fn=collate_fn)
    
    u_review = np.load(os.path.join(args.dataset_dir, 'user_global.npy'), allow_pickle=True)
    u_p_review = np.load(os.path.join(args.dataset_dir, 'user_like.npy'), allow_pickle=True)
    u_n_review = np.load(os.path.join(args.dataset_dir, 'user_dislike.npy'), allow_pickle=True)
    i_review = np.load(os.path.join(args.dataset_dir, 'item_global.npy'), allow_pickle=True)
    i_p_review = np.load(os.path.join(args.dataset_dir, 'item_like.npy'), allow_pickle=True)
    i_n_review = np.load(os.path.join(args.dataset_dir, 'item_dislike.npy'), allow_pickle=True)

    u_ratings, i_ratings = load_ratings(train_path, num_users, num_items)

    model = Model(
        num_users, num_items, embedding_dim=args.dims, hidden_dim=args.hidden, 
        CL=args.CL, aa=args.aa, 
        reviews=[u_review,i_review,u_p_review,i_p_review,u_n_review,i_n_review], 
        ratings=[u_ratings, i_ratings],
        edge_ratio=args.edge_ratio,
        device=device
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    model_name = f'LETTER_{args.rating_threshold}'

    updated = 0
    best = 100
    for epoch in range(args.num_epochs):
        train_loss, mse, cl, aa = train(model, train_loader, optimizer, criterion, device, epoch)
        val_loss, val_mse, val_mae = evaluate_rmse(model, val_loader, device)
        print(f'Epoch {epoch+1}/{args.num_epochs}, Train Loss: {train_loss:.4f}, MSE: {mse:.4f}, Val RMSE: {val_loss:.4f}, MSE: {val_mse:.4f}, MAE: {val_mae:.4f}')
        if best > val_mse:
            best = val_mse
            best_path = os.path.join(args.output_dir, f'{dataset}_{model_name}_{epoch}_b{args.batch}_h{args.hidden}_l{args.lr}')
            best_model = copy.deepcopy(model.state_dict())
            updated = 0

        updated = updated + 1
        if updated > args.early_stop and epoch > 0:
            break
    
    save_model(best_model, best_path)
    model.load_state_dict(best_model)

    test_rmse, test_mse, test_mae = evaluate_rmse(model, test_loader, device)
    print(f'Test RMSE: {test_rmse:.4f}, MSE: {test_mse:.4f}, MAE: {test_mae:.4f}')
    results = {
        'test_rmse': test_rmse,
        'test_mse': test_mse,
        'test_mae': test_mae
    }
    with open(os.path.join(args.output_dir, f'{dataset}_{model_name}_results_{args.seed}.json'), 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='args')
    parser.add_argument('--dataset', default='Toys14', type=str)
    parser.add_argument('--dataset_dir', default='./dataset', type=str)
    parser.add_argument('--output_dir', default='./saved_model', type=str)

    parser.add_argument('--dims', default=768, type=int)
    parser.add_argument('--hidden', default=128, type=int)
    parser.add_argument('--CL', default=0, type=int)
    parser.add_argument('--aa', default=0, type=int)
    parser.add_argument('--cl_lr', default=1, type=float)
    parser.add_argument('--aa_lr', default=1, type=float)
    parser.add_argument('--rating_threshold', default=3, type=int)
    parser.add_argument('--edge_ratio', default=50, type=int)

    parser.add_argument('--batch', default=128, type=int)
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--num_epochs', default=500, type=int)
    parser.add_argument('--reg_lr', default=1, type=float)
    parser.add_argument('--early_stop', default=15, type=int)

    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()

    main()
