import torch
import torch.nn as nn
import torch.nn.functional as F


class DAML(nn.Module):
    """Deep Cooperative Neural Networks with Dual-Attentive Matching (KDD'19 DAML)."""

    def __init__(self, args):
        super().__init__()
        self.args = args

        self.user_word_embs = nn.Embedding(args.vocab_size, args.word_dim)
        self.item_word_embs = nn.Embedding(args.vocab_size, args.word_dim)

        # Shared word-level attention conv (Eq. 1-7 in original implementation notes)
        self.word_cnn = nn.Conv2d(1, 1, (5, args.word_dim), padding=(2, 0))

        # Document-level CNN
        self.user_doc_cnn = nn.Conv2d(1, args.filters_num, (args.kernel_size, args.word_dim), padding=(1, 0))
        self.item_doc_cnn = nn.Conv2d(1, args.filters_num, (args.kernel_size, args.word_dim), padding=(1, 0))

        # Abstract-level CNN
        self.user_abs_cnn = nn.Conv2d(1, args.filters_num, (args.kernel_size, args.filters_num))
        self.item_abs_cnn = nn.Conv2d(1, args.filters_num, (args.kernel_size, args.filters_num))

        self.unfold = nn.Unfold((3, args.filters_num), padding=(1, 0))

        self.user_fc = nn.Linear(args.filters_num, args.id_emb_size)
        self.item_fc = nn.Linear(args.filters_num, args.id_emb_size)

        self.uid_embedding = nn.Embedding(args.user_num + 1, args.id_emb_size)
        self.iid_embedding = nn.Embedding(args.item_num + 1, args.id_emb_size)

        # DAML runner prediction head (keeps behavior close to original framework + LFM default)
        self.pred = nn.Linear(args.id_emb_size * 4, 1)
        self.user_bias = nn.Embedding(args.user_num + 1, 1)
        self.item_bias = nn.Embedding(args.item_num + 1, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout(args.dropout)

        self._reset_parameters(args)

    def _reset_parameters(self, args):
        for cnn in [self.word_cnn, self.user_doc_cnn, self.item_doc_cnn, self.user_abs_cnn, self.item_abs_cnn]:
            nn.init.xavier_normal_(cnn.weight)
            nn.init.uniform_(cnn.bias, -0.1, 0.1)

        for fc in [self.user_fc, self.item_fc]:
            nn.init.uniform_(fc.weight, -0.1, 0.1)
            nn.init.constant_(fc.bias, 0.1)

        nn.init.uniform_(self.uid_embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.iid_embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.pred.weight, -0.1, 0.1)
        nn.init.constant_(self.pred.bias, 0.0)
        nn.init.uniform_(self.user_bias.weight, -0.1, 0.1)
        nn.init.uniform_(self.item_bias.weight, -0.1, 0.1)

        if args.w2v_matrix is not None:
            w2v = torch.from_numpy(args.w2v_matrix)
            self.user_word_embs.weight.data.copy_(w2v)
            self.item_word_embs.weight.data.copy_(w2v)
        else:
            nn.init.uniform_(self.user_word_embs.weight, -0.1, 0.1)
            nn.init.uniform_(self.item_word_embs.weight, -0.1, 0.1)

    def local_attention_cnn(self, word_embs: torch.Tensor, doc_cnn: nn.Conv2d) -> torch.Tensor:
        local_att_words = self.word_cnn(word_embs.unsqueeze(1))
        local_word_weight = torch.sigmoid(local_att_words.squeeze(1))
        word_embs = word_embs * local_word_weight
        return doc_cnn(word_embs.unsqueeze(1))

    def local_pooling_cnn(
        self,
        feature: torch.Tensor,
        attention: torch.Tensor,
        cnn: nn.Conv2d,
        fc: nn.Linear,
    ) -> torch.Tensor:
        bs, n_filters, doc_len, _ = feature.shape
        feature = feature.permute(0, 3, 2, 1)  # [B, 1, doc_len, n_filters]
        attention = attention.reshape(bs, 1, doc_len, 1)

        pools = feature * attention
        pools = self.unfold(pools)
        pools = pools.reshape(bs, 3, n_filters, doc_len)
        pools = pools.sum(dim=1, keepdim=True)
        pools = pools.transpose(2, 3)  # [B, 1, doc_len, n_filters]

        abs_fea = cnn(pools).squeeze(3)
        abs_fea = F.avg_pool1d(abs_fea, abs_fea.size(2))
        abs_fea = F.relu(fc(abs_fea.squeeze(2)))
        return abs_fea

    def forward(
        self,
        user_doc: torch.Tensor,
        item_doc: torch.Tensor,
        uids: torch.Tensor,
        iids: torch.Tensor,
    ) -> torch.Tensor:
        user_word_embs = self.user_word_embs(user_doc)
        item_word_embs = self.item_word_embs(item_doc)

        user_local_fea = self.local_attention_cnn(user_word_embs, self.user_doc_cnn)
        item_local_fea = self.local_attention_cnn(item_word_embs, self.item_doc_cnn)

        # Cross-document attentive matching matrix
        euclidean = (user_local_fea - item_local_fea.permute(0, 1, 3, 2)).pow(2).sum(1).sqrt()
        attention_matrix = 1.0 / (1 + euclidean)
        user_attention = attention_matrix.sum(2)
        item_attention = attention_matrix.sum(1)

        user_doc_fea = self.local_pooling_cnn(user_local_fea, user_attention, self.user_abs_cnn, self.user_fc)
        item_doc_fea = self.local_pooling_cnn(item_local_fea, item_attention, self.item_abs_cnn, self.item_fc)

        uid_emb = self.uid_embedding(uids)
        iid_emb = self.iid_embedding(iids)

        joint = torch.cat([user_doc_fea, uid_emb, item_doc_fea, iid_emb], dim=1)
        joint = self.dropout(joint)

        out = self.pred(joint)
        out = out + self.user_bias(uids) + self.item_bias(iids) + self.global_bias
        return out.squeeze(1)
