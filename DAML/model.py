import torch
import torch.nn as nn
import torch.nn.functional as F


class FM(nn.Module):
    def __init__(self, dim, user_num, item_num):
        super().__init__()
        self.fc = nn.Linear(dim, 1)
        self.fm_v = nn.Parameter(torch.randn(dim, 10))
        self.b_users = nn.Parameter(torch.randn(user_num + 2, 1))
        self.b_items = nn.Parameter(torch.randn(item_num + 2, 1))

    def forward(self, feature, uids, iids):
        linear = self.fc(feature)
        inter1 = torch.mm(feature, self.fm_v).pow(2)
        inter2 = torch.mm(feature.pow(2), self.fm_v.pow(2))
        fm_out = 0.5 * torch.sum(inter1 - inter2, dim=1, keepdim=True) + linear
        return fm_out + self.b_users[uids] + self.b_items[iids]


class LFM(nn.Module):
    def __init__(self, dim, user_num, item_num):
        super().__init__()
        self.fc = nn.Linear(dim, 1)
        self.b_users = nn.Parameter(torch.randn(user_num + 2, 1))
        self.b_items = nn.Parameter(torch.randn(item_num + 2, 1))

    def forward(self, feature, user_id, item_id):
        return self.fc(feature) + self.b_users[user_id] + self.b_items[item_id]


class NFM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, 1)
        self.fm_v = nn.Parameter(torch.randn(16, dim))
        self.mlp = nn.Linear(16, 16)
        self.h = nn.Linear(16, 1, bias=False)
        self.drop = nn.Dropout(0.5)

    def forward(self, x, *args):
        linear = self.fc(x)
        inter1 = torch.mm(x, self.fm_v.t()).pow(2)
        inter2 = torch.mm(x.pow(2), self.fm_v.pow(2).t())
        bilinear = 0.5 * (inter1 - inter2)
        out = F.relu(self.mlp(bilinear))
        out = self.drop(out)
        return self.h(out) + linear


class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, 1)

    def forward(self, feature, *args):
        return F.relu(self.fc(feature))


class PredictionLayer(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.output = opt.output
        if opt.output == "fm":
            self.model = FM(opt.feature_dim, opt.user_num, opt.item_num)
        elif opt.output == "lfm":
            self.model = LFM(opt.feature_dim, opt.user_num, opt.item_num)
        elif opt.output == "mlp":
            self.model = MLP(opt.feature_dim)
        elif opt.output == "nfm":
            self.model = NFM(opt.feature_dim)
        else:
            self.model = None

    def forward(self, feature, uid, iid):
        if self.output in {"lfm", "fm", "nfm"}:
            return self.model(feature, uid, iid)
        if self.output == "mlp":
            return self.model(feature)
        return torch.sum(feature, 1, keepdim=True)


class SelfAtt(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(dim, num_heads, 128, 0.4)
        self.encoder = nn.TransformerEncoder(enc_layer, 1)

    def forward(self, user_fea, item_fea):
        fea = torch.cat([user_fea, item_fea], 1).permute(1, 0, 2)
        out = self.encoder(fea)
        return out.permute(1, 0, 2)


class FusionLayer(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        if opt.self_att:
            self.attn = SelfAtt(opt.id_emb_size, opt.num_heads)

    def forward(self, u_out, i_out):
        if self.opt.self_att:
            out = self.attn(u_out, i_out)
            s_u_out, s_i_out = torch.split(out, out.size(1) // 2, 1)
            u_out = u_out + s_u_out
            i_out = i_out + s_i_out

        if self.opt.r_id_merge == 'cat':
            u_out = u_out.reshape(u_out.size(0), -1)
            i_out = i_out.reshape(i_out.size(0), -1)
        else:
            u_out = u_out.sum(1)
            i_out = i_out.sum(1)

        if self.opt.ui_merge == 'cat':
            return torch.cat([u_out, i_out], 1)
        if self.opt.ui_merge == 'add':
            return u_out + i_out
        return u_out * i_out


class DAML(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.num_fea = 2

        self.user_word_embs = nn.Embedding(opt.vocab_size, opt.word_dim)
        self.item_word_embs = nn.Embedding(opt.vocab_size, opt.word_dim)

        self.word_cnn = nn.Conv2d(1, 1, (5, opt.word_dim), padding=(2, 0))
        self.user_doc_cnn = nn.Conv2d(1, opt.filters_num, (opt.kernel_size, opt.word_dim), padding=(1, 0))
        self.item_doc_cnn = nn.Conv2d(1, opt.filters_num, (opt.kernel_size, opt.word_dim), padding=(1, 0))
        self.user_abs_cnn = nn.Conv2d(1, opt.filters_num, (opt.kernel_size, opt.filters_num))
        self.item_abs_cnn = nn.Conv2d(1, opt.filters_num, (opt.kernel_size, opt.filters_num))

        self.unfold = nn.Unfold((3, opt.filters_num), padding=(1, 0))
        self.user_fc = nn.Linear(opt.filters_num, opt.id_emb_size)
        self.item_fc = nn.Linear(opt.filters_num, opt.id_emb_size)

        self.uid_embedding = nn.Embedding(opt.user_num + 2, opt.id_emb_size)
        self.iid_embedding = nn.Embedding(opt.item_num + 2, opt.id_emb_size)

        self.reset_para()

    def forward(self, datas):
        _, _, uids, iids, _, _, user_doc, item_doc = datas

        user_word_embs = self.user_word_embs(user_doc)
        item_word_embs = self.item_word_embs(item_doc)

        user_local_fea = self.local_attention_cnn(user_word_embs, self.user_doc_cnn)
        item_local_fea = self.local_attention_cnn(item_word_embs, self.item_doc_cnn)

        euclidean = (user_local_fea - item_local_fea.permute(0, 1, 3, 2)).pow(2).sum(1).sqrt()
        attention_matrix = 1.0 / (1 + euclidean)
        user_attention = attention_matrix.sum(2)
        item_attention = attention_matrix.sum(1)

        user_doc_fea = self.local_pooling_cnn(user_local_fea, user_attention, self.user_abs_cnn, self.user_fc)
        item_doc_fea = self.local_pooling_cnn(item_local_fea, item_attention, self.item_abs_cnn, self.item_fc)

        uid_emb = self.uid_embedding(uids)
        iid_emb = self.iid_embedding(iids)

        user_fea = torch.stack([user_doc_fea, uid_emb], 1)
        item_fea = torch.stack([item_doc_fea, iid_emb], 1)
        return user_fea, item_fea

    def local_attention_cnn(self, word_embs, doc_cnn):
        local_att_words = self.word_cnn(word_embs.unsqueeze(1))
        local_word_weight = torch.sigmoid(local_att_words.squeeze(1))
        word_embs = word_embs * local_word_weight
        return doc_cnn(word_embs.unsqueeze(1))

    def local_pooling_cnn(self, feature, attention, cnn, fc):
        bs, n_filters, doc_len, _ = feature.shape
        feature = feature.permute(0, 3, 2, 1)
        attention = attention.reshape(bs, 1, doc_len, 1)
        pools = feature * attention
        pools = self.unfold(pools)
        pools = pools.reshape(bs, 3, n_filters, doc_len)
        pools = pools.sum(dim=1, keepdims=True)
        pools = pools.transpose(2, 3)

        abs_fea = cnn(pools).squeeze(3)
        abs_fea = F.avg_pool1d(abs_fea, abs_fea.size(2))
        abs_fea = F.relu(fc(abs_fea.squeeze(2)))
        return abs_fea

    def reset_para(self):
        for cnn in [self.word_cnn, self.user_doc_cnn, self.item_doc_cnn, self.user_abs_cnn, self.item_abs_cnn]:
            nn.init.xavier_normal_(cnn.weight)
            nn.init.uniform_(cnn.bias, -0.1, 0.1)
        for fc in [self.user_fc, self.item_fc]:
            nn.init.uniform_(fc.weight, -0.1, 0.1)
            nn.init.constant_(fc.bias, 0.1)
        nn.init.uniform_(self.uid_embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.iid_embedding.weight, -0.1, 0.1)

        if self.opt.use_word_embedding and self.opt.w2v_matrix is not None:
            w2v = torch.from_numpy(self.opt.w2v_matrix)
            self.user_word_embs.weight.data.copy_(w2v)
            self.item_word_embs.weight.data.copy_(w2v)


class RecommenderModel(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.net = DAML(opt)

        if opt.ui_merge == 'cat':
            if opt.r_id_merge == 'cat':
                feature_dim = opt.id_emb_size * self.net.num_fea * 2
            else:
                feature_dim = opt.id_emb_size * 2
        else:
            if opt.r_id_merge == 'cat':
                feature_dim = opt.id_emb_size * self.net.num_fea
            else:
                feature_dim = opt.id_emb_size

        self.opt.feature_dim = feature_dim
        self.fusion_net = FusionLayer(opt)
        self.predict_net = PredictionLayer(opt)
        self.dropout = nn.Dropout(opt.drop_out)

    def forward(self, datas):
        user_feature, item_feature = self.net(datas)
        ui_feature = self.fusion_net(user_feature, item_feature)
        ui_feature = self.dropout(ui_feature)
        uids = datas[2]
        iids = datas[3]
        return self.predict_net(ui_feature, uids, iids).squeeze(1)
