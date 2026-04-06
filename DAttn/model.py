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


class LocalAttention(nn.Module):
    def __init__(self, seq_len, win_size, emb_size, filters_num):
        super().__init__()
        self.att_conv = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(win_size, emb_size), padding=((win_size - 1) // 2, 0)),
            nn.Sigmoid(),
        )
        self.cnn = nn.Conv2d(1, filters_num, kernel_size=(1, emb_size))

    def forward(self, x):
        score = self.att_conv(x.unsqueeze(1)).squeeze(1)
        out = x.mul(score)
        out = out.unsqueeze(1)
        out = torch.tanh(self.cnn(out)).squeeze(3)
        out = F.max_pool1d(out, out.size(2)).squeeze(2)
        return out


class GlobalAttention(nn.Module):
    def __init__(self, seq_len, emb_size, filters_size=[2, 3, 4], filters_num=100):
        super().__init__()
        self.att_conv = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(seq_len, emb_size)),
            nn.Sigmoid(),
        )
        self.convs = nn.ModuleList([nn.Conv2d(1, filters_num, (k, emb_size)) for k in filters_size])

    def forward(self, x):
        x = x.unsqueeze(1)
        score = self.att_conv(x)
        x = x.mul(score)
        conv_outs = [torch.tanh(cnn(x).squeeze(3)) for cnn in self.convs]
        conv_outs = [F.max_pool1d(out, out.size(2)).squeeze(2) for out in conv_outs]
        return conv_outs


class D_ATTN(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.num_fea = 1
        self.user_net = _DattnNet(opt)
        self.item_net = _DattnNet(opt)

    def forward(self, datas):
        _, _, _, _, _, _, user_doc, item_doc = datas
        u_fea = self.user_net(user_doc)
        i_fea = self.item_net(item_doc)
        return u_fea, i_fea


class _DattnNet(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.word_embs = nn.Embedding(opt.vocab_size, opt.word_dim)
        self.local_att = LocalAttention(opt.doc_len, win_size=5, emb_size=opt.word_dim, filters_num=opt.filters_num)
        self.global_att = GlobalAttention(opt.doc_len, emb_size=opt.word_dim, filters_num=opt.filters_num)

        fea_dim = opt.filters_num * 4
        self.fc = nn.Sequential(
            nn.Linear(fea_dim, fea_dim),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(fea_dim, opt.id_emb_size),
        )
        self.dropout = nn.Dropout(opt.drop_out)
        self.reset_para()

    def forward(self, docs):
        docs = self.word_embs(docs)
        local_fea = self.local_att(docs)
        global_fea = self.global_att(docs)
        r_fea = torch.cat([local_fea] + global_fea, 1)
        r_fea = self.dropout(r_fea)
        r_fea = self.fc(r_fea)
        return torch.stack([r_fea], 1)

    def reset_para(self):
        cnns = [self.local_att.cnn, self.local_att.att_conv[0]]
        for cnn in cnns:
            nn.init.xavier_uniform_(cnn.weight, gain=1)
            nn.init.uniform_(cnn.bias, -0.1, 0.1)
        for cnn in self.global_att.convs:
            nn.init.xavier_uniform_(cnn.weight, gain=1)
            nn.init.uniform_(cnn.bias, -0.1, 0.1)
        nn.init.uniform_(self.fc[0].weight, -0.1, 0.1)
        nn.init.uniform_(self.fc[-1].weight, -0.1, 0.1)
        if self.opt.use_word_embedding and self.opt.w2v_matrix is not None:
            self.word_embs.weight.data.copy_(torch.from_numpy(self.opt.w2v_matrix))
        else:
            nn.init.xavier_normal_(self.word_embs.weight)


class RecommenderModel(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.net = D_ATTN(opt)

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
