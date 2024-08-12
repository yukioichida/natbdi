import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionWiseFF(nn.Module):

    def __init__(self, embedding_dim:int, dropout_rate=0.1):
        super().__init__()
        self.c_fc = nn.Linear(embedding_dim, 1 * embedding_dim, bias=False)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(1 * embedding_dim, embedding_dim, bias=False)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class BeliefTransformerBlock(nn.Module):

    def __init__(self,
                 belief_dim: int,
                 scale_attention: bool = True,
                 dropout_rate: float = 0.0):
        super(BeliefTransformerBlock, self).__init__()
        self.scale_attention = scale_attention
        self.attention_dropout = nn.Dropout(dropout_rate)
        self.output_dropout = nn.Dropout(dropout_rate)

        self.layer_norm_1 = nn.LayerNorm(belief_dim, bias=False)
        self.qkv_proj_layer = nn.Linear(belief_dim, belief_dim * 3, bias=False)
        self.mlp = PositionWiseFF(belief_dim, dropout_rate)
        self.layer_norm_2 = nn.LayerNorm(belief_dim, bias=False)

    def self_attention(self,
                       x: torch.Tensor,
                       belief_base_sizes: list[int]) -> torch.Tensor:
        # TODO: evaluate using flash attention cuda kernels (pytorch implementation -> F.scaled_dot_product)
        q, k, v = self.qkv_proj_layer(x).chunk(3, dim=-1)
        attention = torch.matmul(q, k.transpose(-1, -2))
        if self.scale_attention:
            attention = attention / math.sqrt(v.size(-1))
        pad_mask = self.mask_padding(attention, belief_base_sizes)
        attention[pad_mask] = -torch.inf
        attention = self.attention_dropout(attention)
        attention = nn.Softmax(dim=-1)(attention)

        y = torch.matmul(attention, v)
        y = self.output_dropout(y)
        return y

    def mask_padding(self, x: torch.Tensor, belief_base_sizes: list[int]) -> torch.Tensor:
        mask = torch.zeros_like(x)
        for i, size in enumerate(belief_base_sizes):
            mask[i, :, size:] = 1
        return mask.bool()

    def forward(self, x: torch.Tensor, belief_base_sizes: list[int]) -> torch.Tensor:
        x = self.layer_norm_1(x)
        x = self.self_attention(x, belief_base_sizes)
        x = self.layer_norm_2(x)
        x = self.mlp(x)
        return x


class BeliefBaseEncoder(nn.Module):

    def __init__(self, belief_dim: int, n_blocks: int = 10):
        super(BeliefBaseEncoder, self).__init__()
        self.blocks = nn.ModuleList([BeliefTransformerBlock(belief_dim=belief_dim) for _ in range(n_blocks)])

    def pooling_belief_base(self, x: torch.Tensor) -> torch.Tensor:
        representation = x[:, 0, :]
        return representation

    def forward(self, x: torch.Tensor, belief_base_sizes: list[int]) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, belief_base_sizes)
        representation = self.pooling_belief_base(x)
        return representation


class QNetwork(nn.Module):

    def __init__(self, action_dim: int, belief_base_dim: int, goal_dim: int, n_blocks: int = 3):
        super(QNetwork, self).__init__()
        self.belief_base_encoder = BeliefBaseEncoder(belief_dim=belief_base_dim, n_blocks=n_blocks)
        # Goal will be in belief base, we test using goal as input but all actions will receive the same goal and,
        # hence, the q-values would be similar regardless of reward
        output_dim = action_dim + belief_base_dim

        self.hidden = nn.Linear(output_dim, belief_base_dim, bias=False)
        self.q_value_layer = nn.Linear(belief_base_dim, 1, bias=False)

    def forward(self,
                belief_base: torch.Tensor,
                belief_base_sizes: list[int],
                action_tensors: torch.Tensor) -> torch.Tensor:
        encoded_belief_base = self.belief_base_encoder(belief_base, belief_base_sizes)  # [bs, belief_dim]
        # encoded_belief_base = encoded_belief_base.unsqueeze(1).repeat(1, num_actions, 1)  # [bs, num_action, belief_dim]
        output_repr = torch.cat([encoded_belief_base, action_tensors], dim=-1)
        x = self.hidden(output_repr)
        x = F.relu(x)
        q_values = self.q_value_layer(x)
        return q_values
