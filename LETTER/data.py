import json
import torch
from torch.utils.data import Dataset


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
