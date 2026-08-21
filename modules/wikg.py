import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GlobalAttention, global_max_pool, global_mean_pool


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class WiKG(nn.Module):

    def __init__(
        self,
        input_dim=1024,
        n_classes=2,
        dropout=0.25,
        act="relu",
        inner_dim=512,
        topk=6,
        agg_type="bi-interaction",
        pool="attn",
        mil_norm=None,
        embed_norm_pos=0,
        mil_bias=True,
        mil_cls_bias=True,
        **kwargs,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.inner_dim = inner_dim
        self.topk = topk
        self.agg_type = agg_type
        self.scale = inner_dim**-0.5
        self.mil_norm = mil_norm
        self.embed_norm_pos = embed_norm_pos

        # Normalization Layers (matching ABMIL)
        if mil_norm == "bn":
            self.norm = (
                nn.BatchNorm1d(input_dim)
                if embed_norm_pos == 0
                else nn.BatchNorm1d(inner_dim)
            )
        elif mil_norm == "ln":
            self.norm = (
                nn.LayerNorm(input_dim, bias=mil_bias)
                if embed_norm_pos == 0
                else nn.LayerNorm(inner_dim, bias=mil_bias)
            )
        else:
            self.norm = nn.Identity()

        # Projection Head
        self._fc1 = nn.Sequential(
            nn.Linear(input_dim, inner_dim, bias=mil_bias),
            (
                nn.GELU()
                if act.lower() == "gelu"
                else (
                    nn.LeakyReLU() if act.lower() == "leakyrelu" else nn.ReLU()
                )
            ),
        )

        self.W_head = nn.Linear(inner_dim, inner_dim, bias=mil_bias)
        self.W_tail = nn.Linear(inner_dim, inner_dim, bias=mil_bias)

        if self.agg_type == "gcn":
            self.linear = nn.Linear(inner_dim, inner_dim, bias=mil_bias)
        elif self.agg_type == "sage":
            self.linear = nn.Linear(inner_dim * 2, inner_dim, bias=mil_bias)
        elif self.agg_type == "bi-interaction":
            self.linear1 = nn.Linear(inner_dim, inner_dim, bias=mil_bias)
            self.linear2 = nn.Linear(inner_dim, inner_dim, bias=mil_bias)
        else:
            raise NotImplementedError

        self.activation = (
            nn.GELU() if act.lower() == "gelu" else nn.LeakyReLU()
        )
        self.message_dropout = (
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )

        self.norm1 = nn.LayerNorm(inner_dim, bias=mil_bias)
        self.classifier = nn.Linear(
            inner_dim, n_classes, bias=mil_cls_bias if mil_bias else mil_bias
        )

        # Readout Strategy
        if pool == "mean":
            self.readout = global_mean_pool
            self.att_net = None
        elif pool == "max":
            self.readout = global_max_pool
            self.att_net = None
        else:  # 'attn'
            self.att_net = nn.Sequential(
                nn.Linear(inner_dim, inner_dim // 2, bias=mil_bias),
                nn.LeakyReLU(),
                nn.Linear(inner_dim // 2, 1, bias=mil_bias),
            )
            self.readout = GlobalAttention(self.att_net)

        self.apply(initialize_weights)

    def forward(
        self,
        x,
        return_attn=False,
        return_act=False,
        no_norm=False,
        pos=None,
        **kwargs,
    ):
        # Input Shape Handling
        if isinstance(x, dict):
            x = x.get("feature", x)

        if len(x.size()) == 2:
            x = x.unsqueeze(0)  # [N, C] -> [1, N, C]

        # Early Normalization (matching ABMIL)
        if self.embed_norm_pos == 0 and self.mil_norm is not None:
            if self.mil_norm == "bn":
                x = torch.transpose(x, -1, -2)
                x = self.norm(x)
                x = torch.transpose(x, -1, -2)
            else:
                x = self.norm(x)

        # Feature Embedding Projection
        x = self._fc1(x)  # [B, N, inner_dim]

        if self.embed_norm_pos == 1 and self.mil_norm is not None:
            if self.mil_norm == "bn":
                x = torch.transpose(x, -1, -2)
                x = self.norm(x)
                x = torch.transpose(x, -1, -2)
            else:
                x = self.norm(x)

        act = x.clone()

        # WiKG Neighbor Construction & Aggregation
        x = (x + x.mean(dim=1, keepdim=True)) * 0.5
        e_h = self.W_head(x)
        e_t = self.W_tail(x)

        # Graph kNN TopK Search
        attn_logit = (e_h * self.scale) @ e_t.transpose(-2, -1)
        k_val = min(self.topk, attn_logit.size(-1))  # Safeguard for small bags
        topk_weight, topk_index = torch.topk(attn_logit, k=k_val, dim=-1)

        topk_index = topk_index.to(torch.long)
        topk_index_expanded = topk_index.expand(e_t.size(0), -1, -1)
        batch_indices = (
            torch.arange(e_t.size(0)).view(-1, 1, 1).to(topk_index.device)
        )

        Nb_h = e_t[batch_indices, topk_index_expanded, :]
        topk_prob = F.softmax(topk_weight, dim=2)

        eh_r = torch.mul(
            topk_prob.unsqueeze(-1), Nb_h
        ) + torch.matmul(
            (1 - topk_prob).unsqueeze(-1), e_h.unsqueeze(2)
        )

        # Gated Knowledge Attention
        e_h_expand = e_h.unsqueeze(2).expand(-1, -1, k_val, -1)
        gate = torch.tanh(e_h_expand + eh_r)
        ka_weight = torch.einsum("ijkl,ijkm->ijk", Nb_h, gate)

        ka_prob = F.softmax(ka_weight, dim=2).unsqueeze(dim=2)
        e_Nh = torch.matmul(ka_prob, Nb_h).squeeze(dim=2)

        # Message Passing Aggregation
        if self.agg_type == "gcn":
            embedding = e_h + e_Nh
            embedding = self.activation(self.linear(embedding))
        elif self.agg_type == "sage":
            embedding = torch.cat([e_h, e_Nh], dim=2)
            embedding = self.activation(self.linear(embedding))
        elif self.agg_type == "bi-interaction":
            sum_embedding = self.activation(self.linear1(e_h + e_Nh))
            bi_embedding = self.activation(self.linear2(e_h * e_Nh))
            embedding = sum_embedding + bi_embedding

        h = self.message_dropout(embedding)

        # Readout / Pooling Layer
        B, N, C = h.shape
        h_flat = h.view(B * N, C)
        batch_idx = (
            torch.arange(B, device=h.device).repeat_interleave(N).long()
        )

        # Compute bag-level attention scores for returns
        if self.att_net is not None:
            attn_scores = self.att_net(h_flat).squeeze(-1).view(B, N)
            A = F.softmax(attn_scores, dim=-1)
        else:
            A = torch.full(
                (B, N), 1.0 / N, device=h.device
            )  # Default uniform attention

        if isinstance(self.readout, GlobalAttention):
            pooled = self.readout(h_flat, batch=batch_idx)
        else:
            pooled = self.readout(h_flat, batch=batch_idx)

        pooled = self.norm1(pooled)
        logits = self.classifier(pooled)

        # Format Return 
        if return_attn:
            output = [logits, A]
            if return_act:
                output.append(act)
            return output
        else:
            return logits