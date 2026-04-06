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


class Co_Attention(nn.Module):
    def __init__(self, dim, gumbel, pooling):
        super().__init__()
        self.gumbel = gumbel
        self.pooling = pooling
        self.M = nn.Parameter(torch.randn(dim, dim))
        self.fc_u = nn.Linear(dim, dim)
        self.fc_i = nn.Linear(dim, dim)

    def forward(self, u_fea, i_fea):
        u = self.fc_u(u_fea)
        i = self.fc_i(i_fea)
        S = u.matmul(self.M).bmm(i.permute(0, 2, 1))
        if self.pooling == 'max':
            u_score = S.max(2)[0]
            i_score = S.max(1)[0]
        else:
            u_score = S.mean(2)
            i_score = S.mean(1)

        if self.gumbel:
            p_u = F.gumbel_softmax(u_score, hard=True, dim=1)
            p_i = F.gumbel_softmax(i_score, hard=True, dim=1)
        else:
            p_u = F.softmax(u_score, dim=1)
            p_i = F.softmax(i_score, dim=1)

        return p_u.unsqueeze(2), p_i.unsqueeze(2)


class MPCN(nn.Module):
    def __init__(self, opt, head=3):
        super().__init__()
        self.opt = opt
        self.num_fea = 1
        self.head = head

        self.user_word_embs = nn.Embedding(opt.vocab_size, opt.word_dim)
        self.item_word_embs = nn.Embedding(opt.vocab_size, opt.word_dim)

        self.fc_g1 = nn.Linear(opt.word_dim, opt.word_dim)
        self.fc_g2 = nn.Linear(opt.word_dim, opt.word_dim)

        self.review_coatt = nn.ModuleList([Co_Attention(opt.word_dim, gumbel=True, pooling='max') for _ in range(head)])
        self.word_coatt = nn.ModuleList([Co_Attention(opt.word_dim, gumbel=False, pooling='avg') for _ in range(head)])

        self.u_fc = self.fc_layer(opt)
        self.i_fc = self.fc_layer(opt)
        self.drop_out = nn.Dropout(opt.drop_out)
        self.reset_para()

    def fc_layer(self, opt):
        return nn.Sequential(
            nn.Linear(opt.word_dim * self.head, opt.word_dim),
            nn.ReLU(),
            nn.Linear(opt.word_dim, opt.id_emb_size),
        )

    def forward(self, datas):
        user_reviews, item_reviews, _, _, _, _, _, _ = datas

        u_word_embs = self.user_word_embs(user_reviews)
        i_word_embs = self.item_word_embs(item_reviews)
        u_reviews = self.review_gate(u_word_embs)
        i_reviews = self.review_gate(i_word_embs)

        u_fea, i_fea = [], []
        for idx in range(self.head):
            r_coatt = self.review_coatt[idx]
            w_coatt = self.word_coatt[idx]
            p_u, p_i = r_coatt(u_reviews, i_reviews)

            u_r_words = user_reviews.permute(0, 2, 1).float().bmm(p_u)
            i_r_words = item_reviews.permute(0, 2, 1).float().bmm(p_i)
            u_words = self.user_word_embs(u_r_words.squeeze(2).long())
            i_words = self.item_word_embs(i_r_words.squeeze(2).long())

            p_u, p_i = w_coatt(u_words, i_words)
            u_w_fea = u_words.permute(0, 2, 1).bmm(p_u).squeeze(2)
            # keep original behavior from Neu-Review-Rec implementation
            i_w_fea = u_words.permute(0, 2, 1).bmm(p_i).squeeze(2)
            u_fea.append(u_w_fea)
            i_fea.append(i_w_fea)

        u_fea = torch.cat(u_fea, 1)
        i_fea = torch.cat(i_fea, 1)
        u_fea = self.drop_out(self.u_fc(u_fea))
        i_fea = self.drop_out(self.i_fc(i_fea))

        return torch.stack([u_fea], 1), torch.stack([i_fea], 1)

    def review_gate(self, reviews):
        reviews = reviews.sum(2)
        return torch.sigmoid(self.fc_g1(reviews)) * torch.tanh(self.fc_g2(reviews))

    def reset_para(self):
        for fc in [self.fc_g1, self.fc_g2, self.u_fc[0], self.u_fc[-1], self.i_fc[0], self.i_fc[-1]]:
            nn.init.uniform_(fc.weight, -0.1, 0.1)
            nn.init.uniform_(fc.bias, -0.1, 0.1)
        if self.opt.use_word_embedding and self.opt.w2v_matrix is not None:
            w2v = torch.from_numpy(self.opt.w2v_matrix)
            self.user_word_embs.weight.data.copy_(w2v)
            self.item_word_embs.weight.data.copy_(w2v)


class RecommenderModel(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.net = MPCN(opt, head=opt.mpcn_head)

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
