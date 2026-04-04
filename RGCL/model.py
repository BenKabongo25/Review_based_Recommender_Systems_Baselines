import dgl.function as fn
import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn
from torch.nn import init


def to_etype_name(rating):
    return str(rating).replace('.', '_')


class GCMCGraphConv(nn.Module):

    def __init__(
        self,
        in_feats,
        out_feats,
        device=None,
        dropout_rate=0.0
    ):
        super(GCMCGraphConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.device = device
        self.dropout = nn.Dropout(dropout_rate)

        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))

        self.prob_score = nn.Linear(out_feats, 1, bias=False)
        self.review_score = nn.Linear(out_feats, 1, bias=False)
        self.review_w = nn.Linear(out_feats, out_feats, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight)
        init.xavier_uniform_(self.prob_score.weight)
        init.xavier_uniform_(self.review_score.weight)
        init.xavier_uniform_(self.review_w.weight)

    def forward(self, graph, feat, weight=None):
        with graph.local_scope():
            feat = self.weight

            graph.srcdata['h'] = feat
            review_feat = graph.edata['review_feat']

            graph.edata['pa'] = torch.sigmoid(self.prob_score(review_feat))
            graph.edata['rf'] = self.review_w(review_feat) * torch.sigmoid(self.review_score(review_feat))
            graph.update_all(lambda edges: {'m': (edges.src['h'] * edges.data['pa'] + edges.data['rf'])
                                                 * self.dropout(edges.src['cj'])},
                             fn.sum(msg='m', out='h'))

            rst = graph.dstdata['h'] * graph.dstdata['ci']

        return rst 


class GCMCLayer(nn.Module):

    def __init__(
        self,
        rating_vals,
        user_in_units,
        item_in_units,
        out_units,
        dropout_rate=0.0,
        device=None
    ):
        super(GCMCLayer, self).__init__()
        self.rating_vals = rating_vals
        self.ufc = nn.Linear(out_units, out_units)
        self.ifc = nn.Linear(out_units, out_units)
        self.dropout = nn.Dropout(dropout_rate)
        sub_conv = {}
        for rating in rating_vals:
            rating = to_etype_name(rating)
            rev_rating = 'rev-%s' % rating
            sub_conv[rating] = GCMCGraphConv(user_in_units,
                                             out_units,
                                             device=device,
                                             dropout_rate=dropout_rate)
            sub_conv[rev_rating] = GCMCGraphConv(item_in_units,
                                                 out_units,
                                                 device=device,
                                                 dropout_rate=dropout_rate)
        self.conv = dglnn.HeteroGraphConv(sub_conv, aggregate='sum')
        self.agg_act = nn.GELU()
        self.device = device
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, graph, ufeat=None, ifeat=None):
        in_feats = {'user': ufeat, 'item': ifeat}
        out_feats = self.conv(graph, in_feats)
        ufeat = out_feats['user']
        ifeat = out_feats['item']
        ufeat = ufeat.view(ufeat.shape[0], -1)
        ifeat = ifeat.view(ifeat.shape[0], -1)

        ufeat = self.agg_act(ufeat)
        ifeat = self.agg_act(ifeat)
        ufeat = self.dropout(ufeat)
        ifeat = self.dropout(ifeat)
        ufeat = self.ufc(ufeat)
        ifeat = self.ifc(ifeat)
        return ufeat, ifeat


class ContrastLoss(nn.Module):

    def __init__(self, feat_size):
        super(ContrastLoss, self).__init__()
        self.w = nn.Parameter(torch.Tensor(feat_size, feat_size))
        init.xavier_uniform_(self.w.data)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, x, y, y_neg=None):
        scores = (x @ self.w * y ).sum(1)
        labels = scores.new_ones(scores.shape)
        pos_loss = self.bce_loss(scores, labels)

        if y_neg is None:
            idx = torch.randperm(y.shape[0])
            y_neg = y[idx, :]
        neg2_scores = (x @ self.w * y_neg).sum(1)
        neg2_labels = neg2_scores.new_zeros(neg2_scores.shape)
        neg2_loss = self.bce_loss(neg2_scores, neg2_labels)

        loss = pos_loss + neg2_loss
        return loss


class MLPPredictorMI(nn.Module):

    def __init__(
        self,
        in_units,
        num_classes,
        dropout_rate=0.0,
        neg_sample_size=1
    ):
        super(MLPPredictorMI, self).__init__()
        self.neg_sample_size = neg_sample_size
        self.dropout = nn.Dropout(dropout_rate)

        self.contrast_loss = ContrastLoss(in_units)

        self.linear = nn.Sequential(
            nn.Linear(in_units * 2, in_units, bias=False),
            nn.ReLU(),
            nn.Linear(in_units, in_units, bias=False),
        )
        self.predictor = nn.Linear(in_units, num_classes, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def neg_sampling(graph):
        review_feat = graph.edata['review_feat']
        neg_review_feat = review_feat[torch.randperm(review_feat.shape[0]), :]
        return neg_review_feat

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        h_fea = self.linear(torch.cat([h_u, h_v], dim=1))
        score = self.predictor(h_fea).squeeze()

        if 'neg_review_feat' in edges.data:
            review_feat = edges.data['review_feat']
            neg_review_feat = edges.data['neg_review_feat']
            mi_score = self.contrast_loss(h_fea, review_feat, neg_review_feat)
            return {'score': score, 'mi_score': mi_score}
        else:
            return {'score': score}

    def forward(self, graph, ufeat, ifeat, cal_edge_mi=True):
        graph.nodes['user'].data['h'] = ufeat
        graph.nodes['item'].data['h'] = ifeat

        if ('review_feat' in graph.edata) & cal_edge_mi:
            graph.edata['neg_review_feat'] = self.neg_sampling(graph)
        else:
            del graph.edata['neg_review_feat']

        with graph.local_scope():
            graph.apply_edges(self.apply_edges)
            if 'mi_score' in graph.edata:
                return graph.edata['score'], graph.edata['mi_score']
            else:
                return graph.edata['score']


class RGCL(nn.Module):
    def __init__(self, params):
        super(RGCL, self).__init__()
        self._params = params
        self.encoder = GCMCLayer(params.rating_vals,
                                 params.src_in_units,
                                 params.dst_in_units,
                                 params.gcn_out_units,
                                 dropout_rate=params.gcn_dropout,
                                 device=params.device)

        if params.train_classification:
            self.decoder = MLPPredictorMI(in_units=params.gcn_out_units,
                                        num_classes=len(params.rating_vals))
        else: 
            self.decoder = MLPPredictorMI(in_units=params.gcn_out_units,
                                        num_classes=1)
        self.contrast_loss = ContrastLoss(params.gcn_out_units)

    def forward(self, enc_graph, dec_graph, ufeat, ifeat, cal_edge_mi=True):

        user_out, item_out = self.encoder(enc_graph, ufeat, ifeat)
        if self._params.distributed:
            user_out = user_out.to(self._params.device)
            item_out = item_out.to(self._params.device)

        if cal_edge_mi:
            pred_ratings, mi_score = self.decoder(dec_graph, user_out, item_out, cal_edge_mi)
            return pred_ratings, mi_score, user_out, item_out
        else:
            pred_ratings = self.decoder(dec_graph, user_out, item_out, cal_edge_mi)
            return pred_ratings, user_out, item_out
