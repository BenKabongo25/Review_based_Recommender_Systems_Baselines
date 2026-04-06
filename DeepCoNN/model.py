import torch
import torch.nn as nn
import torch.nn.functional as F


class FactorizationMachine(nn.Module):
    """FM layer used as the shared interaction layer in DeepCoNN (Eq. 7)."""

    def __init__(self, input_dim: int, user_num: int, item_num: int, fm_rank: int = 10):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.fm_v = nn.Parameter(torch.randn(input_dim, fm_rank))
        self.user_bias = nn.Parameter(torch.zeros(user_num + 1, 1))
        self.item_bias = nn.Parameter(torch.zeros(item_num + 1, 1))

        nn.init.uniform_(self.linear.weight, -0.05, 0.05)
        nn.init.constant_(self.linear.bias, 0.0)
        nn.init.uniform_(self.fm_v, -0.05, 0.05)

    def forward(self, x: torch.Tensor, uids: torch.Tensor, iids: torch.Tensor) -> torch.Tensor:
        linear_term = self.linear(x)

        inter_1 = torch.mm(x, self.fm_v).pow(2)
        inter_2 = torch.mm(x.pow(2), self.fm_v.pow(2))
        fm_term = 0.5 * torch.sum(inter_1 - inter_2, dim=1, keepdim=True)

        return linear_term + fm_term + self.user_bias[uids] + self.item_bias[iids]


class DeepCoNNEncoder(nn.Module):
    """One DeepCoNN encoder branch (user or item): Embedding -> CNN -> max-pool -> FC."""

    def __init__(self, vocab_size: int, word_dim: int, filters_num: int, kernel_size: int, latent_dim: int, dropout: float):
        super().__init__()
        self.word_emb = nn.Embedding(vocab_size, word_dim)
        self.conv = nn.Conv2d(1, filters_num, (kernel_size, word_dim))
        self.fc = nn.Linear(filters_num, latent_dim)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_normal_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0.1)
        nn.init.uniform_(self.fc.weight, -0.1, 0.1)
        nn.init.constant_(self.fc.bias, 0.1)

    def forward(self, doc_ids: torch.Tensor) -> torch.Tensor:
        emb = self.word_emb(doc_ids)
        feat = F.relu(self.conv(emb.unsqueeze(1))).squeeze(3)
        feat = F.max_pool1d(feat, feat.size(2)).squeeze(2)
        feat = self.dropout(self.fc(feat))
        return feat


class DeepCoNN(nn.Module):
    """DeepCoNN model with FM interaction head as described in the paper."""

    def __init__(self, args):
        super().__init__()

        self.user_encoder = DeepCoNNEncoder(
            vocab_size=args.vocab_size,
            word_dim=args.word_dim,
            filters_num=args.filters_num,
            kernel_size=args.kernel_size,
            latent_dim=args.latent_dim,
            dropout=args.dropout,
        )
        self.item_encoder = DeepCoNNEncoder(
            vocab_size=args.vocab_size,
            word_dim=args.word_dim,
            filters_num=args.filters_num,
            kernel_size=args.kernel_size,
            latent_dim=args.latent_dim,
            dropout=args.dropout,
        )

        if args.w2v_matrix is not None:
            w2v = torch.from_numpy(args.w2v_matrix)
            self.user_encoder.word_emb.weight.data.copy_(w2v)
            self.item_encoder.word_emb.weight.data.copy_(w2v)
        else:
            nn.init.uniform_(self.user_encoder.word_emb.weight, -0.1, 0.1)
            nn.init.uniform_(self.item_encoder.word_emb.weight, -0.1, 0.1)

        self.fm = FactorizationMachine(
            input_dim=args.latent_dim * 2,
            user_num=args.user_num,
            item_num=args.item_num,
            fm_rank=10,
        )

    def forward(self, user_doc: torch.Tensor, item_doc: torch.Tensor, uids: torch.Tensor, iids: torch.Tensor) -> torch.Tensor:
        user_vec = self.user_encoder(user_doc)
        item_vec = self.item_encoder(item_doc)
        joint = torch.cat([user_vec, item_vec], dim=1)
        return self.fm(joint, uids, iids).squeeze(1)
