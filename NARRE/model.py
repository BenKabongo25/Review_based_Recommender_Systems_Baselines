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


class _NarreNet(nn.Module):
    def __init__(self, opt, uori='user'):
        super().__init__()
        self.opt = opt

        if uori == 'user':
            id_num = opt.user_num
            ui_id_num = opt.item_num
        else:
            id_num = opt.item_num
            ui_id_num = opt.user_num

        self.id_embedding = nn.Embedding(id_num + 2, opt.id_emb_size)
        self.word_embs = nn.Embedding(opt.vocab_size, opt.word_dim)
        self.u_i_id_embedding = nn.Embedding(ui_id_num + 2, opt.id_emb_size)

        self.review_linear = nn.Linear(opt.filters_num, opt.id_emb_size)
        self.id_linear = nn.Linear(opt.id_emb_size, opt.id_emb_size, bias=False)
        self.attention_linear = nn.Linear(opt.id_emb_size, 1)
        self.fc_layer = nn.Linear(opt.filters_num, opt.id_emb_size)
        self.cnn = nn.Conv2d(1, opt.filters_num, (opt.kernel_size, opt.word_dim))
        self.dropout = nn.Dropout(opt.drop_out)
        self.reset_para()

    def forward(self, reviews, ids, ids_list):
        reviews = self.word_embs(reviews)
        _, r_num, r_len, wd = reviews.size()
        reviews = reviews.view(-1, r_len, wd)

        id_emb = self.id_embedding(ids)
        u_i_id_emb = self.u_i_id_embedding(ids_list)

        fea = F.relu(self.cnn(reviews.unsqueeze(1))).squeeze(3)
        fea = F.max_pool1d(fea, fea.size(2)).squeeze(2)
        fea = fea.view(-1, r_num, fea.size(1))

        rs_mix = F.relu(self.review_linear(fea) + self.id_linear(F.relu(u_i_id_emb)))
        att_score = self.attention_linear(rs_mix)
        att_weight = F.softmax(att_score, 1)
        r_fea = (fea * att_weight).sum(1)
        r_fea = self.dropout(r_fea)

        return torch.stack([id_emb, self.fc_layer(r_fea)], 1)

    def reset_para(self):
        nn.init.xavier_normal_(self.cnn.weight)
        nn.init.constant_(self.cnn.bias, 0.1)
        nn.init.uniform_(self.id_linear.weight, -0.1, 0.1)
        nn.init.uniform_(self.review_linear.weight, -0.1, 0.1)
        nn.init.constant_(self.review_linear.bias, 0.1)
        nn.init.uniform_(self.attention_linear.weight, -0.1, 0.1)
        nn.init.constant_(self.attention_linear.bias, 0.1)
        nn.init.uniform_(self.fc_layer.weight, -0.1, 0.1)
        nn.init.constant_(self.fc_layer.bias, 0.1)
        if self.opt.use_word_embedding and self.opt.w2v_matrix is not None:
            self.word_embs.weight.data.copy_(torch.from_numpy(self.opt.w2v_matrix))
        else:
            nn.init.xavier_normal_(self.word_embs.weight)
        nn.init.uniform_(self.id_embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.u_i_id_embedding.weight, -0.1, 0.1)


class NARRE(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.num_fea = 2
        self.user_net = _NarreNet(opt, 'user')
        self.item_net = _NarreNet(opt, 'item')

    def forward(self, datas):
        user_reviews, item_reviews, uids, iids, user_item2id, item_user2id, _, _ = datas
        u_fea = self.user_net(user_reviews, uids, user_item2id)
        i_fea = self.item_net(item_reviews, iids, item_user2id)
        return u_fea, i_fea


class RecommenderModel(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.net = NARRE(opt)

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
