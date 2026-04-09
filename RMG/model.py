import torch
import torch.nn as nn


class AttnPool(nn.Module):
    """Tanh attention pooling used repeatedly in RMG (word/sentence/review/neighbor levels)."""

    def __init__(self, in_dim: int, attn_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, attn_dim)
        self.score = nn.Linear(attn_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        attn = torch.tanh(self.proj(x))
        logits = self.score(attn).squeeze(-1)
        weight = torch.softmax(logits, dim=1)
        return torch.bmm(weight.unsqueeze(1), x).squeeze(1)


class HierDocEncoder(nn.Module):
    """
    Hierarchical text encoder close to the paper implementation:
    words -> sentence vectors -> review vectors -> entity vector.
    """

    def __init__(
        self,
        vocab_size: int,
        word_dim: int,
        cnn_filters: int,
        cnn_window: int,
        attn_dim: int,
        dropout: float,
        w2v_matrix=None,
    ):
        super().__init__()
        self.word_emb = nn.Embedding(vocab_size, word_dim, padding_idx=0)
        self.word_conv = nn.Conv1d(word_dim, cnn_filters, kernel_size=cnn_window, padding=cnn_window // 2)
        self.sent_conv = nn.Conv1d(cnn_filters, cnn_filters, kernel_size=cnn_window, padding=cnn_window // 2)

        self.word_pool = AttnPool(cnn_filters, attn_dim)
        self.sent_pool = AttnPool(cnn_filters, attn_dim)
        self.review_pool = AttnPool(cnn_filters, attn_dim)

        self.dropout = nn.Dropout(dropout)

        if w2v_matrix is not None:
            self.word_emb.weight.data.copy_(torch.from_numpy(w2v_matrix))
        else:
            nn.init.uniform_(self.word_emb.weight, -0.05, 0.05)
            with torch.no_grad():
                self.word_emb.weight[0].zero_()

        nn.init.xavier_uniform_(self.word_conv.weight)
        nn.init.constant_(self.word_conv.bias, 0.0)
        nn.init.xavier_uniform_(self.sent_conv.weight)
        nn.init.constant_(self.sent_conv.bias, 0.0)

    def forward(self, docs: torch.Tensor) -> torch.Tensor:
        # docs: [B, R, S, W]
        bsz, n_reviews, n_sents, n_words = docs.shape

        words = docs.reshape(bsz * n_reviews * n_sents, n_words)
        wemb = self.word_emb(words)  # [B*R*S, W, Dw]
        wfea = torch.relu(self.word_conv(wemb.transpose(1, 2))).transpose(1, 2)  # [B*R*S, W, F]
        wfea = self.dropout(wfea)

        sent_vec = self.word_pool(wfea)  # [B*R*S, F]
        sent_vec = sent_vec.reshape(bsz * n_reviews, n_sents, -1)

        sfea = torch.relu(self.sent_conv(sent_vec.transpose(1, 2))).transpose(1, 2)  # [B*R, S, F]
        sfea = self.dropout(sfea)
        review_vec = self.sent_pool(sfea)  # [B*R, F]

        review_vec = review_vec.reshape(bsz, n_reviews, -1)
        review_vec = self.dropout(review_vec)
        entity_vec = self.review_pool(review_vec)  # [B, F]
        return entity_vec


class RMG(nn.Module):
    """
    Reviews Meet Graphs model.
    - Text tower: hierarchical attentive CNN encoder.
    - Graph tower: two-hop attentive aggregation on user-item bipartite graph.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

        self.text_user = HierDocEncoder(
            vocab_size=args.vocab_size,
            word_dim=args.word_dim,
            cnn_filters=args.cnn_filters,
            cnn_window=args.cnn_window,
            attn_dim=args.attn_dim,
            dropout=args.dropout,
            w2v_matrix=args.w2v_matrix,
        )
        self.text_item = HierDocEncoder(
            vocab_size=args.vocab_size,
            word_dim=args.word_dim,
            cnn_filters=args.cnn_filters,
            cnn_window=args.cnn_window,
            attn_dim=args.attn_dim,
            dropout=args.dropout,
            w2v_matrix=args.w2v_matrix,
        )

        # Paper setting: user/item id embedding = 100.
        self.user_id_emb = nn.Embedding(args.user_num + 1, args.id_emb_dim, padding_idx=args.user_num)
        self.item_id_emb = nn.Embedding(args.item_num + 1, args.id_emb_dim, padding_idx=args.item_num)

        self.user_nei_pool = AttnPool(args.id_emb_dim, args.attn_dim)
        self.item_nei_pool = AttnPool(args.id_emb_dim, args.attn_dim)
        self.user_2hop_pool = AttnPool(args.id_emb_dim * 2, args.attn_dim)
        self.item_2hop_pool = AttnPool(args.id_emb_dim * 2, args.attn_dim)

        self.dropout = nn.Dropout(args.dropout)
        # factor_u and factor_i: [text(150), id(100), graph(200)] = 450 when paper defaults are used.
        self.pred = nn.Linear(args.cnn_filters + args.id_emb_dim + args.id_emb_dim * 2, 1)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.uniform_(self.user_id_emb.weight, -0.05, 0.05)
        nn.init.uniform_(self.item_id_emb.weight, -0.05, 0.05)
        with torch.no_grad():
            self.user_id_emb.weight[self.args.user_num].zero_()
            self.item_id_emb.weight[self.args.item_num].zero_()

        nn.init.xavier_uniform_(self.pred.weight)
        nn.init.constant_(self.pred.bias, 0.0)

    def forward(
        self,
        user_docs: torch.Tensor,
        item_docs: torch.Tensor,
        user_neighbors: torch.Tensor,
        item_neighbors: torch.Tensor,
        user_item_user: torch.Tensor,
        item_user_item: torch.Tensor,
        uids: torch.Tensor,
        iids: torch.Tensor,
    ) -> torch.Tensor:
        user_text = self.text_user(user_docs)
        item_text = self.text_item(item_docs)

        # 1-hop neighborhood aggregation.
        u_item_nei = self.item_id_emb(user_neighbors)  # [B, N, D]
        i_user_nei = self.user_id_emb(item_neighbors)  # [B, N, D]
        # 2-hop neighborhood aggregation.
        u_2hop_users = self.user_id_emb(user_item_user)  # [B, N, N, D]
        i_2hop_items = self.item_id_emb(item_user_item)  # [B, N, N, D]

        u_2hop_users = u_2hop_users.reshape(-1, u_2hop_users.size(2), u_2hop_users.size(3))
        i_2hop_items = i_2hop_items.reshape(-1, i_2hop_items.size(2), i_2hop_items.size(3))
        u_2hop_flat = self.user_nei_pool(self.dropout(u_2hop_users)).reshape(user_neighbors.size(0), user_neighbors.size(1), -1)
        i_2hop_flat = self.item_nei_pool(self.dropout(i_2hop_items)).reshape(item_neighbors.size(0), item_neighbors.size(1), -1)

        u_graph = self.user_2hop_pool(torch.cat([u_item_nei, u_2hop_flat], dim=-1))
        i_graph = self.item_2hop_pool(torch.cat([i_user_nei, i_2hop_flat], dim=-1))

        uid_vec = self.user_id_emb(uids)
        iid_vec = self.item_id_emb(iids)

        factor_u = torch.cat([user_text, uid_vec, u_graph], dim=1)
        factor_i = torch.cat([item_text, iid_vec, i_graph], dim=1)

        joint = factor_u * factor_i
        joint = self.dropout(joint)
        # Keras baseline used Dense(1, relu) on element-wise product.
        pred = torch.relu(self.pred(joint)).squeeze(1)
        return pred
