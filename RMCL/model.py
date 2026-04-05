from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMCL(nn.Module):
    """Review-based Multi-intention Contrastive Learning (RMCL).

    Implementation follows the paper equations:
    - Intention mixture and expectation (Eq. 6-9)
    - Similarity and orthogonality constraints (Eq. 10-11)
    - Intention contrastive objective (Eq. 14)
    - User-item matching and rating regression (Eq. 15-17)
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        text_dim: int,
        latent_dim: int = 128,
        num_intentions: int = 20,
        dropout: float = 0.2,
        lambda_cl: float = 1.0,
        eta_sim: float = 1.0,
        mu_ind: float = 1.0,
    ):
        super().__init__()
        self.num_intentions = num_intentions
        self.lambda_cl = lambda_cl
        self.eta_sim = eta_sim
        self.mu_ind = mu_ind

        # Implicit intention vectors h_i in Eq. (6).
        self.intentions = nn.Parameter(torch.randn(num_intentions, text_dim) * 0.02)

        # Encoder(c_u) -> pi_u in Eq. (7), 3-layer MLP + softmax.
        self.intent_encoder = nn.Sequential(
            nn.Linear(text_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, num_intentions),
        )

        # Feature encoders for x_u and x_i in Eq. (15).
        self.user_embedding = nn.Embedding(num_users, latent_dim)
        self.item_embedding = nn.Embedding(num_items, latent_dim)
        self.user_feature_encoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.item_feature_encoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # q_u and q_i are in text_dim; project to latent for matching.
        self.intent_projector = nn.Sequential(
            nn.Linear(text_dim, latent_dim),
            nn.ReLU(),
        )

        # Eq. (15): f([q_u ◦ q_i, q_u - q_i, e_u, e_i]).
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim * 4, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, 1),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def _intention_distribution(self, text_repr: torch.Tensor) -> torch.Tensor:
        logits = self.intent_encoder(text_repr)
        return F.softmax(logits, dim=-1)

    def _comprehensive_intent(self, pi: torch.Tensor) -> torch.Tensor:
        # Eq. (9): q = sum_i pi_i h_i
        return torch.matmul(pi, self.intentions)

    def _similarity_loss(self, q: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # Eq. (10) is maximization of cosine similarity.
        # Convert to minimization loss as 1 - cosine.
        cos = F.cosine_similarity(q, c, dim=-1)
        return (1.0 - cos).mean()

    def _orthogonal_loss(self) -> torch.Tensor:
        # Eq. (11) encourages independent intentions.
        # We penalize off-diagonal elements of H H^T.
        h = F.normalize(self.intentions, dim=-1)
        gram = torch.matmul(h, h.t())
        eye = torch.eye(self.num_intentions, device=gram.device)
        off_diag = gram * (1.0 - eye)
        return (off_diag ** 2).sum()

    def _contrastive_loss(self, c: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
        # Eq. (14): weighted contrastive objective over intentions.
        c_n = F.normalize(c, dim=-1)
        h_n = F.normalize(self.intentions, dim=-1)
        logits = torch.matmul(c_n, h_n.t())
        log_probs = F.log_softmax(logits, dim=-1)
        return (-(pi * log_probs).sum(dim=-1)).mean()

    def forward(
        self,
        user_ids: torch.LongTensor,
        item_ids: torch.LongTensor,
        user_text_repr: torch.Tensor,
        item_text_repr: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # E-step style intention distributions (Eq. 8)
        pi_u = self._intention_distribution(user_text_repr)
        pi_i = self._intention_distribution(item_text_repr)

        q_u = self._comprehensive_intent(pi_u)
        q_i = self._comprehensive_intent(pi_i)

        q_u_lat = self.intent_projector(q_u)
        q_i_lat = self.intent_projector(q_i)

        e_u = self.user_feature_encoder(self.user_embedding(user_ids))
        e_i = self.item_feature_encoder(self.item_embedding(item_ids))

        # Eq. (15)
        pair_feat = torch.cat([q_u_lat * q_i_lat, q_u_lat - q_i_lat, e_u, e_i], dim=-1)
        pred = self.predictor(pair_feat).squeeze(-1)

        aux = {
            "pi_u": pi_u,
            "pi_i": pi_i,
            "q_u": q_u,
            "q_i": q_i,
            "user_text": user_text_repr,
            "item_text": item_text_repr,
        }
        return pred, aux

    def compute_loss(
        self,
        pred_ratings: torch.Tensor,
        true_ratings: torch.Tensor,
        aux: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        # Eq. (16)
        l_rate = F.mse_loss(pred_ratings, true_ratings)

        # Eq. (14), both user and item side.
        l_cl_u = self._contrastive_loss(aux["user_text"], aux["pi_u"])
        l_cl_i = self._contrastive_loss(aux["item_text"], aux["pi_i"])
        l_cl = 0.5 * (l_cl_u + l_cl_i)

        # Eq. (10) similarity constraints for user/item.
        l_sim_u = self._similarity_loss(aux["q_u"], aux["user_text"])
        l_sim_i = self._similarity_loss(aux["q_i"], aux["item_text"])
        l_sim = 0.5 * (l_sim_u + l_sim_i)

        # Eq. (11)
        l_ind = self._orthogonal_loss()

        # Eq. (17)
        total = l_rate + self.lambda_cl * l_cl + self.eta_sim * l_sim + self.mu_ind * l_ind
        return {
            "loss": total,
            "rate": l_rate,
            "cl": l_cl,
            "sim": l_sim,
            "ind": l_ind,
        }
