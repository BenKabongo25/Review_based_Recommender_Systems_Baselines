import torch
import torch.nn as nn
import torch.nn.functional as F


class NarreEncoder(nn.Module):
    """NARRE encoder branch for user or item side."""

    def __init__(self, args, side: str):
        super().__init__()
        self.args = args

        if side == "user":
            id_num = args.user_num
            ui_id_num = args.item_num
        elif side == "item":
            id_num = args.item_num
            ui_id_num = args.user_num
        else:
            raise ValueError(f"Unknown side: {side}")

        self.id_embedding = nn.Embedding(id_num + 1, args.latent_dim)
        self.word_emb = nn.Embedding(args.vocab_size, args.word_dim)
        self.ui_id_embedding = nn.Embedding(ui_id_num + 1, args.latent_dim)

        self.cnn = nn.Conv2d(1, args.filters_num, (args.kernel_size, args.word_dim))

        self.review_linear = nn.Linear(args.filters_num, args.latent_dim)
        self.id_linear = nn.Linear(args.latent_dim, args.latent_dim, bias=False)
        self.attention_linear = nn.Linear(args.latent_dim, 1)

        self.fc = nn.Linear(args.filters_num, args.latent_dim)
        self.dropout = nn.Dropout(args.dropout)

        self._reset_parameters(args)

    def _reset_parameters(self, args):
        nn.init.xavier_normal_(self.cnn.weight)
        nn.init.constant_(self.cnn.bias, 0.1)

        nn.init.uniform_(self.id_linear.weight, -0.1, 0.1)

        nn.init.uniform_(self.review_linear.weight, -0.1, 0.1)
        nn.init.constant_(self.review_linear.bias, 0.1)

        nn.init.uniform_(self.attention_linear.weight, -0.1, 0.1)
        nn.init.constant_(self.attention_linear.bias, 0.1)

        nn.init.uniform_(self.fc.weight, -0.1, 0.1)
        nn.init.constant_(self.fc.bias, 0.1)

        if args.w2v_matrix is not None:
            self.word_emb.weight.data.copy_(torch.from_numpy(args.w2v_matrix))
        else:
            nn.init.xavier_normal_(self.word_emb.weight)

        nn.init.uniform_(self.id_embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.ui_id_embedding.weight, -0.1, 0.1)

    def forward(self, reviews: torch.Tensor, ids: torch.Tensor, ids_list: torch.Tensor):
        # reviews: [B, R, L]
        reviews = self.word_emb(reviews)
        bsz, r_num, r_len, emb_dim = reviews.size()
        reviews = reviews.view(-1, r_len, emb_dim)

        id_emb = self.id_embedding(ids)  # q_u or p_i
        ui_id_emb = self.ui_id_embedding(ids_list)

        # CNN text processor for each review
        fea = F.relu(self.cnn(reviews.unsqueeze(1))).squeeze(3)
        fea = F.max_pool1d(fea, fea.size(2)).squeeze(2)
        fea = fea.view(-1, r_num, fea.size(1))

        # Attention over reviews (content + counterpart id)
        rs_mix = F.relu(self.review_linear(fea) + self.id_linear(F.relu(ui_id_emb)))
        att_score = self.attention_linear(rs_mix)
        att_weight = F.softmax(att_score, dim=1)

        o_vec = (fea * att_weight).sum(1)
        o_vec = self.dropout(o_vec)  # dropout on review pooling layer (paper Sec. 4.4)

        review_vec = self.fc(o_vec)  # X_u or Y_i
        return id_emb, review_vec


class NARRE(nn.Module):
    """Neural Attentional Rating Regression with Review-level Explanations."""

    def __init__(self, args):
        super().__init__()
        self.user_encoder = NarreEncoder(args, side="user")
        self.item_encoder = NarreEncoder(args, side="item")

        self.pred = nn.Linear(args.latent_dim, 1, bias=False)  # W1^T in Eq. (12)
        self.user_bias = nn.Embedding(args.user_num + 1, 1)
        self.item_bias = nn.Embedding(args.item_num + 1, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout(args.dropout)  # dropout on h0

        nn.init.uniform_(self.pred.weight, -0.1, 0.1)
        nn.init.uniform_(self.user_bias.weight, -0.1, 0.1)
        nn.init.uniform_(self.item_bias.weight, -0.1, 0.1)

    def forward(
        self,
        user_reviews: torch.Tensor,
        item_reviews: torch.Tensor,
        uids: torch.Tensor,
        iids: torch.Tensor,
        user_item_ids: torch.Tensor,
        item_user_ids: torch.Tensor,
    ) -> torch.Tensor:
        q_u, x_u = self.user_encoder(user_reviews, uids, user_item_ids)
        p_i, y_i = self.item_encoder(item_reviews, iids, item_user_ids)

        # Eq. (11): h0 = (q_u + X_u) ⊙ (p_i + Y_i)
        h0 = (q_u + x_u) * (p_i + y_i)
        h0 = self.dropout(h0)

        # Eq. (12): r_hat = W1^T h0 + b_u + b_i + mu
        out = self.pred(h0)
        out = out + self.user_bias(uids) + self.item_bias(iids) + self.global_bias
        return out.squeeze(1)
