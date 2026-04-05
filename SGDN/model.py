import dgl
import dgl.function as fn
import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import init
from torch_scatter import scatter




def to_etype_name(rating):
    return str(rating).replace('.', '_')


def get_activation(act):
    if act is None:
        return lambda x: x
    if isinstance(act, str):
        if act == 'leaky':
            return nn.LeakyReLU(0.1)
        elif act == 'relu':
            return nn.ReLU()
        elif act == 'tanh':
            return nn.Tanh()
        elif act == 'sigmoid':
            return nn.Sigmoid()
        elif act == 'softsign':
            return nn.Softsign()
        else:
            raise NotImplementedError
    else:
        return act


class GCMCGraphConv(nn.Module):

    def __init__(self,
                 in_feats,
                 out_feats,
                 k,
                 num_factor,
                 rating_name_to_idx,
                 device=None,
                 dropout_rate=0.0):
        super(GCMCGraphConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.k = k
        self.device = device
        self.rating_name_to_idx = rating_name_to_idx

        self.dropout = nn.Dropout(dropout_rate)
        self.review_w = nn.Linear(self._out_feats*num_factor, 64//num_factor, bias=False)
        self.node_w = nn.Linear(64//num_factor, 64//num_factor, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.review_w.weight)
        init.xavier_uniform_(self.node_w.weight)

    def forward(self, graph, feat, weight=None):
        with graph.local_scope():
            e = graph.canonical_etypes[0][1]
            rating_name = e[4:] if e.startswith('rev-') else e
            rating_idx = self.rating_name_to_idx[rating_name]
            s = 'h_' + str(rating_idx + 1)
            graph.srcdata['h'] = self.node_w(feat[0][s])
            review_feat = graph.edata['review_feat']
            graph.edata['rf'] = self.review_w(review_feat)
            graph.update_all(lambda edges: {'m': (edges.src['h']
                                                  + edges.data['rf']) * self.dropout(edges.data['w'])},
                             fn.sum(msg='m', out='h'))
            rst = graph.dstdata['h']

        return rst


class GCMCLayer(nn.Module):

    def __init__(self,
                 rating_vals,
                 user_in_units,
                 item_in_units,
                 num_rating,
                 msg_units,
                 out_units,
                 k,
                 num_factor,
                 aggregate='sum',
                 dropout_rate=0.0,
                 device=None):
        super(GCMCLayer, self).__init__()
        self.rating_vals = rating_vals
        self.ufc = nn.Linear(msg_units, out_units)
        self.ifc = nn.Linear(msg_units, out_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.k = k
        sub_conv = {}
        self.aggregate = aggregate  # stack or sum
        self.num_user = user_in_units
        self.num_item = item_in_units
        self.num_factor = num_factor
        self.num_rating = num_rating
        self.rating_name_to_idx = {
            to_etype_name(r): idx for idx, r in enumerate(self.rating_vals)
        }
        self.eta = nn.Parameter(torch.FloatTensor(len(self.rating_vals), self.num_rating))

        for rating in rating_vals:
            rating = to_etype_name(rating)
            rev_rating = 'rev-%s' % rating
            self.W_r = None
            sub_conv[rating] = GCMCGraphConv(user_in_units,
                                             msg_units,
                                             k,
                                             num_factor,
                                             self.rating_name_to_idx,
                                             device=device,
                                             dropout_rate=dropout_rate)
            sub_conv[rev_rating] = GCMCGraphConv(item_in_units,
                                                 msg_units,
                                                 k,
                                                 num_factor,
                                                 self.rating_name_to_idx,
                                                 device=device,
                                                 dropout_rate=dropout_rate)

        self.conv = dglnn.HeteroGraphConv(sub_conv, aggregate=self.aggregate)
        self.agg_act = nn.LeakyReLU(0.1)
        self.device = device
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, graph, prototypes_all, review_feat_dic, ufeat, ifeat, feat_dic):
        tau =0.5
        exp_anchor_dot = {}

        num_user, num_item = self.num_user, self.num_item
        norm_user_sum, norm_item_sum = torch.zeros(num_user).to(self.device), torch.zeros(num_item).to(self.device)
        for u, e, v in graph.canonical_etypes:
            rating_name = e[4:] if e.startswith('rev-') else e
            rating = self.rating_name_to_idx[rating_name]
            e_s, e_sum = 'h_' + str(rating + 1), 'h_sum' + str(rating + 1)
            row, col = graph[(u, e, v)].edges()[0].to(torch.int64), graph[(u, e, v)].edges()[1].to(torch.int64)
            row_feat, col_feat = F.normalize(feat_dic[self.k][u][e_s][row], dim=1), F.normalize(feat_dic[self.k][v][e_s][col], dim=1)
            row_all = feat_dic[u][e_sum][row]
            col_all = feat_dic[v][e_sum][col]
            sim_k = (row_feat*col_feat).sum(1) / tau
            sim_all = (row_all*col_all).sum(2) / tau
            exp_sim = torch.exp(sim_k) / torch.exp(sim_all).sum(1)

            review_feat_all = review_feat_dic[rating_name]
            review_feat = review_feat_dic[rating_name][:, self.k, :]
            graph.edges[e].data['review_feat'] = review_feat
            prototypes = prototypes_all
            anchor_dot_k = torch.matmul(review_feat, prototypes[self.k]) / tau
            anchor_dot_all = (review_feat_all * prototypes).sum(2) / tau

            exp_anchor_dot_k = torch.exp(anchor_dot_k) / torch.exp(anchor_dot_all).sum(1)
            exp_anchor_dot_k = F.sigmoid(self.eta[rating][:exp_anchor_dot_k.shape[0]])*exp_anchor_dot_k + (1-F.sigmoid(self.eta[rating][:exp_anchor_dot_k.shape[0]]))*exp_sim
            exp_anchor_dot[u+e+v] = exp_anchor_dot_k

            if u == 'item':
                norm_item = scatter(exp_anchor_dot_k, row, dim=0, dim_size=num_item, reduce='sum')
                norm_user = scatter(exp_anchor_dot_k, col, dim=0, dim_size=num_user, reduce='sum')
            else:
                norm_user = scatter(exp_anchor_dot_k, row, dim=0, dim_size=num_user, reduce='sum')
                norm_item = scatter(exp_anchor_dot_k, col, dim=0, dim_size=num_item, reduce='sum')
            norm_user_sum += norm_user
            norm_item_sum += norm_item
        norm_user_sum, norm_item_sum = norm_user_sum/2, norm_item_sum/2
        int_dist = []
        for u, e, v in graph.canonical_etypes:
            exp_anchor_dot_k = exp_anchor_dot[u+e+v]

            row, col = graph[(u, e, v)].edges()[0].to(torch.int64), graph[(u, e, v)].edges()[1].to(torch.int64)
            if u=='item':
                n_ij = torch.sqrt(norm_item_sum[row] * norm_user_sum[col])
            else:
                n_ij = torch.sqrt(norm_user_sum[row] * norm_item_sum[col])

            graph.edges[e].data['w'] = (exp_anchor_dot_k/ n_ij).unsqueeze(1)
            if u == 'item':
                int_dist.append(graph.edges[e].data['w'])
        int_dist = torch.cat(int_dist, dim=0)
        out_feats = self.conv(graph, feat_dic[self.k])
        ufeat = out_feats['user']
        ifeat = out_feats['item']

        # fc and non-linear
        ufeat = self.agg_act(ufeat)
        ifeat = self.agg_act(ifeat)
        ufeat = self.dropout(ufeat)
        ifeat = self.dropout(ifeat)
        ufeat = self.ufc(ufeat)
        ifeat = self.ifc(ifeat)
        return ufeat, ifeat, int_dist


def cal_c_loss(h_fea1, h_fea2, int_dist, rating_split, k):
    tau = 0.2
    pos = 0
    c_loss = 0
    for num in rating_split:
        h_fea1_rating = F.normalize(h_fea1[pos: pos + num], dim=1)
        h_fea2_rating = F.normalize(h_fea2[pos: pos + num], dim=1)
        int_dist_rating = int_dist[pos: pos + num]
        sim_matrix = torch.matmul(int_dist_rating, int_dist_rating.transpose(0,1))
        _, indices = torch.topk(sim_matrix, dim=1, k=k)
        pos_fea = h_fea2_rating[indices]
        pos_score = (pos_fea.transpose(0,1)*h_fea1_rating).sum(dim=2).transpose(0, 1)
        pos_score = torch.exp(pos_score / tau).sum(dim=1)

        rand_index = torch.randperm(num, out=None, dtype=torch.int64)[:2048]
        ttl_score = torch.matmul(h_fea1_rating, h_fea2_rating[rand_index].transpose(0, 1))
        ttl_score = torch.sum(torch.exp(ttl_score / tau), axis=1)
        c_loss += - torch.mean(torch.log(pos_score / ttl_score))
        pos += num

    return c_loss


class MLPPredictor(nn.Module):
    def __init__(self,
                 in_units,
                 rating_split,
                 num_classes,
                 num_factor,
                 dropout_rate=0.0):
        super(MLPPredictor, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Sequential(
            nn.Linear(in_units * 2, 64, bias=False),
            nn.GELU(),
            nn.Linear(64, 64, bias=False),
        )
        self.predictor = nn.Linear(64, num_classes, bias=False)
        self.rating_split = rating_split
        self.num_factor = num_factor
        self.reset_parameters()
        self.w = nn.Linear(in_units * 2//num_factor, 1)

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        if self.num_factor==1:
            h_fea = self.linear(torch.cat([h_u, h_v], dim=1))
            score = self.predictor(h_fea)
            return {'score': score, 'feat': h_fea}
        h_u, h_v = h_u.view(h_u.shape[0], self.num_factor, -1), h_v.view(h_v.shape[0], self.num_factor, -1)
        x = []
        for k in range(self.num_factor):
            h_fea_k = torch.cat([h_u[:, k, :], h_v[:, k, :]], dim=1)
            x.append(h_fea_k.unsqueeze(1))
        x = torch.cat(x, dim=1)
        x= x.reshape(x.shape[0], -1)
        h_fea = self.linear(x)
        score = self.predictor(h_fea).squeeze()
        return {'score': score, 'feat': h_fea}

    def forward(self, graph, ufeat, ifeat):
        graph.nodes['item'].data['h'] = ifeat
        graph.nodes['user'].data['h'] = ufeat

        with graph.local_scope():
            graph.apply_edges(self.apply_edges)
            return graph.edata['score'], graph.edata['feat']


class SGDN(nn.Module):

    def __init__(self, params, num_user, num_item, review_feat_dic, num_rating, rating_split):
        super(SGDN, self).__init__()
        self._act = get_activation(params.model_activation)
        self.encoders = [[] for _ in range(params.num_layer)]
        self.num_factor = params.num_factor
        self.num_user = num_user
        self.num_item = num_item
        self.num_rating = num_rating
        self.rating_vals = params.rating_vals
        self.rating_name_to_idx = {
            to_etype_name(r): idx for idx, r in enumerate(self.rating_vals)
        }
        self.rating_split = rating_split
        self.dim = params.gcn_out_units
        self.device = params.device
        self.num_layer = params.num_layer
        self.dropout = nn.Dropout(params.gcn_dropout)
        for l in range(params.num_layer):
            for k in range(params.num_factor):
                if l < params.num_layer-1:
                    aggr = 'stack'
                else:
                    aggr = 'sum'
                self.encoder = GCMCLayer(params.rating_vals,
                                         params.src_in_units,
                                         params.dst_in_units,
                                         self.num_rating,
                                         params.gcn_agg_units//self.num_factor,
                                         params.gcn_out_units//self.num_factor,
                                         k,
                                         self.num_factor,
                                         aggr,
                                         dropout_rate=params.gcn_dropout,
                                         device=params.device).to(params.device)
                self.encoders[l].append(self.encoder)
        for l in range(params.num_layer):
            for i, encoder in enumerate(self.encoders[l]):
                self.add_module('encoder_{}'.format(l*self.num_factor+i), encoder)
        self.ufeats = nn.ParameterDict()
        self.ifeats = nn.ParameterDict()
        for r in range(len(self.rating_vals)):
            for k in range(self.num_factor):
                self.ufeats[str(r*len(self.rating_vals)+k)] = nn.Parameter(torch.Tensor(num_user, params.gcn_out_units//self.num_factor))
        for r in range(len(self.rating_vals)):
            for k in range(self.num_factor):
                self.ifeats[str(r*len(self.rating_vals)+k)] = nn.Parameter(torch.Tensor(num_item, params.gcn_out_units//self.num_factor))
        self.prototypes = nn.Parameter(torch.Tensor(self.num_factor, params.review_feat_size)).to(params.device)
        self.init_prot(review_feat_dic)
        self.rfcs = [nn.Linear(params.review_feat_size, params.review_feat_size).to(params.device) for _ in range(self.num_factor)]
        for i, fc in enumerate(self.rfcs):
            self.add_module('rfc_{}'.format(i), fc)

        self.decoder = MLPPredictor(in_units=params.gcn_out_units, rating_split=self.rating_split,
                                        num_classes=1, num_factor=self.num_factor, dropout_rate=0.0).to(params.device)
        self.reset_parameters()

    def init_prot(self, review_feat_dic):
        reviews = []
        for rating in self.rating_vals:
            rating = to_etype_name(rating)
            review_feat = review_feat_dic[rating]
            reviews.append(review_feat)
        review_feat = torch.cat(reviews, dim=0).detach().to(torch.float32)

        centroids = None
        faiss_mod = None
        try:
            import faiss as faiss_mod  # type: ignore
        except Exception:
            faiss_mod = None

        if faiss_mod is not None:
            try:
                review_np = review_feat.cpu().numpy()
                kmeans = faiss_mod.Kmeans(d=self.dim, k=self.num_factor, gpu=False)
                kmeans.train(review_np)
                centroids = torch.tensor(kmeans.centroids, dtype=torch.float32, device=self.device)
            except Exception:
                centroids = None

        if centroids is None:
            centroids, _ = self._torch_kmeans(review_feat.to(self.device), self.num_factor, n_iter=30)

        centroids = F.normalize(centroids, p=2, dim=1)
        self.prototypes.data = centroids

    @staticmethod
    def _torch_kmeans(x, k, n_iter=30):
        """Fallback K-Means"""
        n = x.shape[0]
        if n < k:
            raise ValueError(f"Number of points ({n}) must be >= number of clusters ({k}).")

        perm = torch.randperm(n, device=x.device)
        centroids = x[perm[:k]].clone()

        for _ in range(n_iter):
            dist = torch.cdist(x, centroids, p=2)
            labels = torch.argmin(dist, dim=1)
            new_centroids = []
            for cid in range(k):
                mask = labels == cid
                if torch.any(mask):
                    new_centroids.append(x[mask].mean(dim=0))
                else:
                    new_centroids.append(x[torch.randint(0, n, (1,), device=x.device)].squeeze(0))
            new_centroids = torch.stack(new_centroids, dim=0)

            if torch.allclose(new_centroids, centroids, atol=1e-4):
                centroids = new_centroids
                break
            centroids = new_centroids

        return centroids, labels

    def reset_parameters(self):
        for r in range(len(self.rating_vals)):
            for k in range(self.num_factor):
                init.xavier_uniform_(self.ufeats[str(r*len(self.rating_vals)+k)])
        for r in range(len(self.rating_vals)):
            for k in range(self.num_factor):
                init.xavier_uniform_(self.ifeats[str(r*len(self.rating_vals)+k)])
        for i, rfeat in enumerate(self.rfcs):
            init.xavier_uniform_(self.rfcs[i].weight)
        init.xavier_uniform_(self.prototypes)

    def prepare_graph(self, l, graphs, user_out, item_out):
        feat_dic_all = {'user': {}, 'item': {} }
        user_sum, item_sum = [[] for _ in range(len(self.rating_vals))], [[] for _ in range(len(self.rating_vals))]
        for k in range(self.num_factor):

            graph = graphs[k]

            dic = {'user': {}, 'item': {}}

            for u, e, v in graph.canonical_etypes:
                rating_name = e[4:] if e.startswith('rev-') else e
                rating = self.rating_name_to_idx[rating_name]
                if u == 'user':
                    s = 'h_' + str(rating+1)
                    if l == 0:
                        dic['user'][s] = self.ufeats[str(rating*len(self.rating_vals)+k)]
                    else:
                        dic['user'][s] = user_out[k][:, rating, :]
                    user_sum[rating].append(dic['user'][s].unsqueeze(1))
                else:
                    s = 'h_' + str(rating+1)
                    if l == 0:
                        dic['item'][s] = self.ifeats[str(rating*len(self.rating_vals)+k)]
                    else:
                        dic['item'][s] = item_out[k][:, rating, :]
                    item_sum[rating].append(dic['item'][s].unsqueeze(1))
            feat_dic_all[k] = dic
        for rating in range(len(self.rating_vals)):
            feat_dic_all['user']['h_sum'+str(rating+1)] = F.normalize(torch.cat(user_sum[rating], dim=1), dim=2)
            feat_dic_all['item']['h_sum'+str(rating+1)] = F.normalize(torch.cat(item_sum[rating], dim=1), dim=2)

        return feat_dic_all

    def forward(self, enc_graphs, dec_graph, review_feat_dic, save_graph=False):
        review_dic_fact = {}
        has_dec_review_feat = 'review_feat' in dec_graph.edata
        for rating in self.rating_vals:
            rating = to_etype_name(rating)
            review_feat = review_feat_dic[rating]
            temp = []
            for k in range(self.num_factor):
                review_feat_k = self._modules['rfc_'+str(k)](review_feat)
                if has_dec_review_feat:
                    dec_graph.edata['review_feat_'+str(k)] = self._modules['rfc_'+str(k)](dec_graph.edata['review_feat'])
                temp.append(review_feat_k.unsqueeze(1))
            review_dic_fact[rating] = torch.cat(temp, dim=1)

        user_emb, item_emb, int_dists, feat_dic_all = [], [], [], {}
        user_out, item_out, user_out_all, item_out_all = [None for _ in range(self.num_factor)], [None for _ in range(
            self.num_factor)], [torch.zeros(self.num_user, self.dim//self.num_factor).to(self.device) for _
                                                            in range(self.num_factor)],[torch.zeros(self.num_item, self.dim//self.num_factor).to(self.device) for _
                                                            in range(self.num_factor)]
        for l in range(self.num_layer):
            feat_dic_all = self.prepare_graph(l, enc_graphs, user_out, item_out)
            for k in range(self.num_factor):
                user_out[k], item_out[k], int_dist = self._modules['encoder_'+str(l*self.num_factor+k)](enc_graphs[k], self.prototypes, review_dic_fact, self.ufeats, self.ifeats, feat_dic_all)
                if l !=self.num_layer-1:
                    user_out_all[k] += torch.sum(user_out[k], dim=1) * (1.0/(self.num_layer))
                    item_out_all[k] += torch.sum(item_out[k], dim=1) * (1.0 / (self.num_layer))
                else:
                    user_out_all[k] += user_out[k] * (1.0 / (self.num_layer))
                    item_out_all[k] += item_out[k] * (1.0 / (self.num_layer))
                    int_dists.append(int_dist)
        int_dists = torch.cat(int_dists, dim=1)

        for k in range(self.num_factor):
            user_emb.append(user_out_all[k])
            item_emb.append(item_out_all[k])
            if k == 0:
                user_out, item_out = user_out_all[k], item_out_all[k]
            else:
                user_out = torch.cat([user_out, user_out_all[k]], dim=1)
                item_out = torch.cat([item_out, item_out_all[k]], dim=1)

        pred_ratings, h_fea = self.decoder(dec_graph, user_out, item_out)
        pred_ratings = pred_ratings.squeeze()

        return pred_ratings, h_fea, int_dists, user_emb, item_emb, user_out, item_out
