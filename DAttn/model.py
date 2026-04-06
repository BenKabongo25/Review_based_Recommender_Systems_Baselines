import torch
import torch.nn as nn
import torch.nn.functional as F


class FactorizationMachine(nn.Module):
    """FM head used after concatenating user/item representations."""

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


class LocalAttention(nn.Module):
    """Local attention branch from D-Attn: attention conv + feature conv."""

    def __init__(self, window_size: int, emb_dim: int, filters_num: int):
        super().__init__()
        self.att_conv = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(window_size, emb_dim), padding=((window_size - 1) // 2, 0)),
            nn.Sigmoid(),
        )
        self.cnn = nn.Conv2d(1, filters_num, kernel_size=(1, emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        score = self.att_conv(x.unsqueeze(1)).squeeze(1)
        out = x * score
        out = torch.tanh(self.cnn(out.unsqueeze(1))).squeeze(3)
        out = F.max_pool1d(out, out.size(2)).squeeze(2)
        return out


class GlobalAttention(nn.Module):
    """Global attention branch from D-Attn with multiple convolution windows."""

    def __init__(self, seq_len: int, emb_dim: int, kernel_sizes: list[int], filters_num: int):
        super().__init__()
        self.att_conv = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(seq_len, emb_dim)),
            nn.Sigmoid(),
        )
        self.convs = nn.ModuleList([nn.Conv2d(1, filters_num, (k, emb_dim)) for k in kernel_sizes])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = x.unsqueeze(1)
        score = self.att_conv(x)
        x = x * score
        outs = [torch.tanh(conv(x).squeeze(3)) for conv in self.convs]
        outs = [F.max_pool1d(out, out.size(2)).squeeze(2) for out in outs]
        return outs


class DAttnEncoder(nn.Module):
    """Single D-Attn encoder (for user or item)."""

    def __init__(self, args):
        super().__init__()
        self.word_emb = nn.Embedding(args.vocab_size, args.word_dim)
        self.local_att = LocalAttention(args.local_window_size, args.word_dim, args.local_filters_num)
        self.global_att = GlobalAttention(
            args.doc_len,
            args.word_dim,
            args.global_kernel_sizes,
            args.global_filters_num,
        )

        fea_dim = args.local_filters_num + args.global_filters_num * len(args.global_kernel_sizes)
        self.fc = nn.Sequential(
            nn.Linear(fea_dim, args.fc_hidden_dim),
            nn.Dropout(args.dropout),
            nn.ReLU(),
            nn.Linear(args.fc_hidden_dim, args.latent_dim),
            nn.Dropout(args.dropout),
            nn.ReLU(),
        )

    def forward(self, doc_ids: torch.Tensor) -> torch.Tensor:
        x = self.word_emb(doc_ids)
        local_feat = self.local_att(x)
        global_feats = self.global_att(x)
        rep = torch.cat([local_feat] + global_feats, dim=1)
        rep = self.fc(rep)
        return rep


class DAttn(nn.Module):
    """D-Attn with dual encoders + FM prediction head."""

    def __init__(self, args):
        super().__init__()
        self.user_encoder = DAttnEncoder(args)
        self.item_encoder = DAttnEncoder(args)
        self.fm = FactorizationMachine(args.latent_dim * 2, args.user_num, args.item_num, fm_rank=10)

    def forward(self, user_doc: torch.Tensor, item_doc: torch.Tensor, uids: torch.Tensor, iids: torch.Tensor) -> torch.Tensor:
        u = self.user_encoder(user_doc)
        i = self.item_encoder(item_doc)
        joint = torch.cat([u, i], dim=1)
        return self.fm(joint, uids, iids).squeeze(1)
