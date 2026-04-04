import dgl
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


def to_etype_name(rating):
    return str(rating).replace('.', '_')


def _load_indices(npy_path: str) -> np.ndarray:
    idx = np.load(npy_path, allow_pickle=True)
    idx = np.asarray(idx).reshape(-1)
    if idx.dtype.kind not in {"i", "u"}:
        raise ValueError(f"Indices must be integer dtype, got {idx.dtype}.")
    return idx


class RGCLDataset:
    def __init__(
        self,
        dataset_csv,
        train_idx_path,
        eval_idx_path,
        test_idx_path,
        review_feat_path,
        device,
        review_fea_size,
        symm=True,
    ):
        self._device = device
        self._review_fea_size = review_fea_size
        self._symm = symm

        self.train_review_feat = self._load_review_feat(review_feat_path)

        df = pd.read_csv(dataset_csv)
        train_idx = _load_indices(train_idx_path)
        eval_idx = _load_indices(eval_idx_path)
        test_idx = _load_indices(test_idx_path)

        for split_name, split_idx in (("train", train_idx), ("eval", eval_idx), ("test", test_idx)):
            if len(split_idx) == 0:
                raise ValueError(f"{split_name} indices are empty.")
            if split_idx.min() < 0 or split_idx.max() >= len(df):
                raise IndexError(
                    f"{split_name} indices out of bounds for dataset of size {len(df)}."
                )

        train_df = df.iloc[train_idx].copy()
        eval_df = df.iloc[eval_idx].copy()
        test_df = df.iloc[test_idx].copy()

        all_df = pd.concat([train_df, eval_df, test_df], axis=0, ignore_index=True)
        self.user2nid = {u: i for i, u in enumerate(all_df["user_id"].astype(str).unique().tolist())}
        self.item2nid = {m: i for i, m in enumerate(all_df["item_id"].astype(str).unique().tolist())}

        self._num_user = len(self.user2nid)
        self._num_item = len(self.item2nid)

        self.train_datas = self._process_split_df(train_df)
        self.valid_datas = self._process_split_df(eval_df)
        self.test_datas = self._process_split_df(test_df)

        self.possible_rating_values = np.unique(self.train_datas[2])

        self.user_feature = None
        self.item_feature = None

        self.user_feature_shape = (self.num_user, self.num_user)
        self.item_feature_shape = (self.num_item, self.num_item)

        train_rating_pairs, train_rating_values, train_ui_raw = self._generate_pair_value("train")
        valid_rating_pairs, valid_rating_values, valid_ui_raw = self._generate_pair_value("valid")
        test_rating_pairs, test_rating_values, test_ui_raw = self._generate_pair_value("test")

        def _make_labels(ratings):
            labels = torch.LongTensor(
                np.searchsorted(self.possible_rating_values, ratings)
            ).to(device)
            return labels

        self.train_enc_graph = self._generate_enc_graph(
            train_rating_pairs,
            train_rating_values,
            train_ui_raw,
            add_support=True,
        )
        self.train_dec_graph = self._generate_dec_graph(
            train_rating_pairs,
            train_ui_raw,
            review_feat=self.train_review_feat,
        )
        self.train_labels = _make_labels(train_rating_values)
        self.train_truths = torch.FloatTensor(train_rating_values).to(device)

        self.valid_enc_graph = self.train_enc_graph
        self.valid_dec_graph = self._generate_dec_graph(
            valid_rating_pairs,
            valid_ui_raw,
            review_feat=None,
        )
        self.valid_labels = _make_labels(valid_rating_values)
        self.valid_truths = torch.FloatTensor(valid_rating_values).to(device)

        self.test_enc_graph = self.train_enc_graph
        self.test_dec_graph = self._generate_dec_graph(
            test_rating_pairs,
            test_ui_raw,
            review_feat=None,
        )
        self.test_labels = _make_labels(test_rating_values)
        self.test_truths = torch.FloatTensor(test_rating_values).to(device)

        def _npairs(graph):
            rst = 0
            for r in self.possible_rating_values:
                r = to_etype_name(r)
                rst += graph.number_of_edges(str(r))
            return rst

        print(
            "Train enc graph: \t#user:{}\t#item:{}\t#pairs:{}".format(
                self.train_enc_graph.number_of_nodes("user"),
                self.train_enc_graph.number_of_nodes("item"),
                _npairs(self.train_enc_graph),
            )
        )
        print(
            "Train dec graph: \t#user:{}\t#item:{}\t#pairs:{}".format(
                self.train_dec_graph.number_of_nodes("user"),
                self.train_dec_graph.number_of_nodes("item"),
                self.train_dec_graph.number_of_edges(),
            )
        )
        print(
            "Valid enc graph: \t#user:{}\t#item:{}\t#pairs:{}".format(
                self.valid_enc_graph.number_of_nodes("user"),
                self.valid_enc_graph.number_of_nodes("item"),
                _npairs(self.valid_enc_graph),
            )
        )
        print(
            "Valid dec graph: \t#user:{}\t#item:{}\t#pairs:{}".format(
                self.valid_dec_graph.number_of_nodes("user"),
                self.valid_dec_graph.number_of_nodes("item"),
                self.valid_dec_graph.number_of_edges(),
            )
        )
        print(
            "Test enc graph: \t#user:{}\t#item:{}\t#pairs:{}".format(
                self.test_enc_graph.number_of_nodes("user"),
                self.test_enc_graph.number_of_nodes("item"),
                _npairs(self.test_enc_graph),
            )
        )
        print(
            "Test dec graph: \t#user:{}\t#item:{}\t#pairs:{}".format(
                self.test_dec_graph.number_of_nodes("user"),
                self.test_dec_graph.number_of_nodes("item"),
                self.test_dec_graph.number_of_edges(),
            )
        )

    @staticmethod
    def _normalize_review_feat_keys(raw_feat):
        norm = {}
        for (u, i), v in raw_feat.items():
            key = (str(u), str(i))
            if not torch.is_tensor(v):
                v = torch.as_tensor(v)
            norm[key] = v.detach().cpu().to(torch.float32)
        return norm

    def _load_review_feat(self, review_feat_path):
        try:
            review_feat = torch.load(review_feat_path, map_location="cpu")
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Review feature file not found: {review_feat_path}"
            ) from exc

        if not isinstance(review_feat, dict):
            raise TypeError("Review feature file must contain a dict keyed by (user_id, item_id).")

        return self._normalize_review_feat_keys(review_feat)

    def _process_split_df(self, split_df: pd.DataFrame):
        user_raw = split_df["user_id"].astype(str).tolist()
        item_raw = split_df["item_id"].astype(str).tolist()
        rating = split_df["rating"].astype(np.float32).tolist()

        user_nid = [self.user2nid[u] for u in user_raw]
        item_nid = [self.item2nid[i] for i in item_raw]

        return user_nid, item_nid, rating, user_raw, item_raw

    def _generate_pair_value(self, sub_dataset):
        if sub_dataset == "all_train":
            user_id = self.train_datas[0] + self.valid_datas[0]
            item_id = self.train_datas[1] + self.valid_datas[1]
            rating = self.train_datas[2] + self.valid_datas[2]
            user_raw = self.train_datas[3] + self.valid_datas[3]
            item_raw = self.train_datas[4] + self.valid_datas[4]
        elif sub_dataset == "train":
            user_id, item_id, rating, user_raw, item_raw = self.train_datas
        elif sub_dataset == "valid":
            user_id, item_id, rating, user_raw, item_raw = self.valid_datas
        else:
            user_id, item_id, rating, user_raw, item_raw = self.test_datas

        rating_pairs = (
            np.array(user_id, dtype=np.int64),
            np.array(item_id, dtype=np.int64),
        )
        rating_values = np.array(rating, dtype=np.float32)
        ui_raw = list(zip(user_raw, item_raw))
        return rating_pairs, rating_values, ui_raw

    def _lookup_review_feat(self, ui_raw):
        missing = []
        feats = []
        for k in ui_raw:
            if k not in self.train_review_feat:
                missing.append(k)
                continue
            feats.append(self.train_review_feat[k])

        if missing:
            example = missing[0]
            raise KeyError(
                f"Missing review embedding for {len(missing)} pairs. Example missing key: {example}"
            )

        return torch.stack(feats).to(torch.float32)

    def _generate_enc_graph(self, rating_pairs, rating_values, ui_raw, add_support=False):
        data_dict = {}
        num_nodes_dict = {"user": self._num_user, "item": self._num_item}
        rating_row, rating_col = rating_pairs

        review_feat_all = self._lookup_review_feat(ui_raw)

        review_data_dict = {}
        for rating in self.possible_rating_values:
            ridx = np.where(rating_values == rating)[0]
            rrow = rating_row[ridx]
            rcol = rating_col[ridx]
            etype = to_etype_name(rating)
            data_dict.update(
                {
                    ("user", str(etype), "item"): (rrow, rcol),
                    ("item", f"rev-{etype}", "user"): (rcol, rrow),
                }
            )
            review_data_dict[str(etype)] = review_feat_all[ridx]

        graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)
        for rating in self.possible_rating_values:
            etype = to_etype_name(rating)
            graph[str(etype)].edata["review_feat"] = review_data_dict[str(etype)]
            graph[f"rev-{etype}"].edata["review_feat"] = review_data_dict[str(etype)]

        assert len(rating_pairs[0]) == sum([graph.number_of_edges(et) for et in graph.etypes]) // 2

        if add_support:
            def _calc_norm(x):
                x = x.numpy().astype("float32")
                x[x == 0.0] = np.inf
                x = torch.FloatTensor(1.0 / np.sqrt(x))
                return x.unsqueeze(1)

            user_ci = []
            user_cj = []
            item_ci = []
            item_cj = []
            for r in self.possible_rating_values:
                r = to_etype_name(r)
                user_ci.append(graph[f"rev-{r}"].in_degrees())
                item_ci.append(graph[r].in_degrees())
                if self._symm:
                    user_cj.append(graph[r].out_degrees())
                    item_cj.append(graph[f"rev-{r}"].out_degrees())
                else:
                    user_cj.append(torch.zeros((self.num_user,)))
                    item_cj.append(torch.zeros((self.num_item,)))
            user_ci = _calc_norm(sum(user_ci))
            item_ci = _calc_norm(sum(item_ci))
            if self._symm:
                user_cj = _calc_norm(sum(user_cj))
                item_cj = _calc_norm(sum(item_cj))
            else:
                user_cj = torch.ones(self.num_user)
                item_cj = torch.ones(self.num_item)
            graph.nodes["user"].data.update({"ci": user_ci, "cj": user_cj})
            graph.nodes["item"].data.update({"ci": item_ci, "cj": item_cj})

        return graph

    def _generate_dec_graph(self, rating_pairs, ui_raw, review_feat=None):
        ones = np.ones_like(rating_pairs[0])
        user_item_ratings_coo = sp.coo_matrix(
            (ones, rating_pairs),
            shape=(self.num_user, self.num_item),
            dtype=np.float32,
        )
        g = dgl.bipartite_from_scipy(user_item_ratings_coo, utype="_U", etype="_E", vtype="_V")
        g = dgl.heterograph(
            {("user", "rate", "item"): g.edges()},
            num_nodes_dict={"user": self.num_user, "item": self.num_item},
        )
        if review_feat is not None:
            feat = self._lookup_review_feat(ui_raw)
            g.edata["review_feat"] = feat

        return g

    @property
    def num_links(self):
        return self.possible_rating_values.size

    @property
    def num_user(self):
        return self._num_user

    @property
    def num_item(self):
        return self._num_item
