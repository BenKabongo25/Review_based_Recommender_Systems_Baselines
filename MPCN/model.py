import torch
import torch.nn as nn
import torch.nn.functional as F


class CoAttention(nn.Module):
    def __init__(self, dim: int, gumbel: bool, pooling: str):
        super().__init__()
        self.gumbel = gumbel
        self.pooling = pooling
        self.M = nn.Parameter(torch.randn(dim, dim))
        self.fc_u = nn.Linear(dim, dim)
        self.fc_i = nn.Linear(dim, dim)

    def forward(self, u_fea: torch.Tensor, i_fea: torch.Tensor):
        u = self.fc_u(u_fea)
        i = self.fc_i(i_fea)
        s = u.matmul(self.M).bmm(i.permute(0, 2, 1))

        if self.pooling == "max":
            u_score = s.max(2)[0]
            i_score = s.max(1)[0]
        else:
            u_score = s.mean(2)
            i_score = s.mean(1)

        if self.gumbel:
            p_u = F.gumbel_softmax(u_score, hard=True, dim=1)
            p_i = F.gumbel_softmax(i_score, hard=True, dim=1)
        else:
            p_u = F.softmax(u_score, dim=1)
            p_i = F.softmax(i_score, dim=1)

        return p_u.unsqueeze(2), p_i.unsqueeze(2)


class FactorizationMachine(nn.Module):
    """FM prediction layer from Eq. (12) in MPCN paper."""

    def __init__(self, dim: int, num_factors: int = 10):
        super().__init__()
        self.w0 = nn.Parameter(torch.zeros(1))
        self.w = nn.Parameter(torch.zeros(dim))
        self.v = nn.Parameter(torch.randn(dim, num_factors) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        linear = self.w0 + torch.sum(self.w * x, dim=1, keepdim=True)
        inter_1 = torch.mm(x, self.v).pow(2)
        inter_2 = torch.mm(x.pow(2), self.v.pow(2))
        inter = 0.5 * torch.sum(inter_1 - inter_2, dim=1, keepdim=True)
        return (linear + inter).squeeze(1)


class MPCN(nn.Module):
    """Multi-Pointer Co-Attention Network (KDD'18)."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_pointers = args.num_pointers

        self.user_word_embs = nn.Embedding(args.vocab_size, args.word_dim)
        self.item_word_embs = nn.Embedding(args.vocab_size, args.word_dim)

        self.fc_g1 = nn.Linear(args.word_dim, args.word_dim)
        self.fc_g2 = nn.Linear(args.word_dim, args.word_dim)

        self.review_coatt = nn.ModuleList(
            [CoAttention(args.word_dim, gumbel=True, pooling="max") for _ in range(self.num_pointers)]
        )
        self.word_coatt = nn.ModuleList(
            [CoAttention(args.word_dim, gumbel=False, pooling="avg") for _ in range(self.num_pointers)]
        )

        self.dropout = nn.Dropout(args.dropout)
        self.user_proj = self._build_proj(args)
        self.item_proj = self._build_proj(args)

        if args.pointer_agg == "concat":
            pred_in_dim = args.word_dim * self.num_pointers * 2
        else:
            pred_in_dim = args.word_dim * 2
        self.fm = FactorizationMachine(pred_in_dim, num_factors=10)

        self._reset_parameters()

    def _build_proj(self, args):
        if args.pointer_agg == "neural":
            return nn.Sequential(
                nn.Linear(args.word_dim * self.num_pointers, args.word_dim),
                nn.ReLU(),
                nn.Linear(args.word_dim, args.word_dim),
            )
        return None

    def _reset_parameters(self):
        for fc in [self.fc_g1, self.fc_g2]:
            nn.init.uniform_(fc.weight, -0.1, 0.1)
            nn.init.uniform_(fc.bias, -0.1, 0.1)

        # Paper reports pretrained embeddings degrade; initialize/train from scratch.
        nn.init.uniform_(self.user_word_embs.weight, -0.1, 0.1)
        nn.init.uniform_(self.item_word_embs.weight, -0.1, 0.1)

        if self.user_proj is not None:
            for layer in [self.user_proj[0], self.user_proj[-1], self.item_proj[0], self.item_proj[-1]]:
                nn.init.uniform_(layer.weight, -0.1, 0.1)
                nn.init.uniform_(layer.bias, -0.1, 0.1)

    def review_gate(self, reviews: torch.Tensor):
        reviews = reviews.sum(2)
        return torch.sigmoid(self.fc_g1(reviews)) * torch.tanh(self.fc_g2(reviews))

    def _aggregate(self, reps: list[torch.Tensor], side: str):
        if self.args.pointer_agg == "add":
            return torch.stack(reps, dim=0).sum(0)

        cat = torch.cat(reps, dim=1)
        if self.args.pointer_agg == "concat":
            return cat

        if side == "user":
            return self.user_proj(cat)
        return self.item_proj(cat)

    def forward(self, user_reviews: torch.Tensor, item_reviews: torch.Tensor):
        u_word_embs = self.user_word_embs(user_reviews)
        i_word_embs = self.item_word_embs(item_reviews)

        u_reviews = self.review_gate(u_word_embs)
        i_reviews = self.review_gate(i_word_embs)

        u_reps, i_reps = [], []
        for idx in range(self.num_pointers):
            r_coatt = self.review_coatt[idx]
            w_coatt = self.word_coatt[idx]

            p_u, p_i = r_coatt(u_reviews, i_reviews)

            u_r_words = user_reviews.permute(0, 2, 1).float().bmm(p_u)
            i_r_words = item_reviews.permute(0, 2, 1).float().bmm(p_i)
            u_words = self.user_word_embs(u_r_words.squeeze(2).long())
            i_words = self.item_word_embs(i_r_words.squeeze(2).long())

            p_u_w, p_i_w = w_coatt(u_words, i_words)
            u_w_fea = u_words.permute(0, 2, 1).bmm(p_u_w).squeeze(2)
            i_w_fea = i_words.permute(0, 2, 1).bmm(p_i_w).squeeze(2)
            u_reps.append(u_w_fea)
            i_reps.append(i_w_fea)

        u_fea = self._aggregate(u_reps, side="user")
        i_fea = self._aggregate(i_reps, side="item")

        u_fea = self.dropout(u_fea)
        i_fea = self.dropout(i_fea)

        joint = torch.cat([u_fea, i_fea], dim=1)
        return self.fm(joint)
