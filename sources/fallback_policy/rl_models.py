import math

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from sources.fallback_policy.encoder import EncoderModel
from sources.fallback_policy.model import BeliefBaseEncoder


class QNetwork(nn.Module):

    def __init__(self, action_dim: int, belief_base_dim: int, n_blocks: int = 3):
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
                action_tensors: torch.Tensor,
                return_attentions: bool = False):
        encoded_belief_base, attention = self.belief_base_encoder(belief_base, belief_base_sizes)  # [bs, belief_dim]
        x = torch.cat([encoded_belief_base, action_tensors], dim=-1)
        x = F.relu(x)
        x = self.hidden(x)
        x = F.relu(x)
        q_values = self.q_value_layer(x)
        # q_values = F.tanh(q_values)

        if return_attentions:
            return q_values, attention
        else:
            return q_values


class SimpleQNetwork(nn.Module):

    def __init__(self, action_dim: int, belief_base_dim: int, n_blocks: int = 3):
        super(SimpleQNetwork, self).__init__()
        self.belief_base_encoder = nn.ModuleList([nn.Linear(belief_base_dim, belief_base_dim)
                                                  for _ in range(n_blocks)])
        output_dim = action_dim + belief_base_dim

        self.hidden = nn.Linear(output_dim, belief_base_dim, bias=False)
        self.q_value_layer = nn.Linear(belief_base_dim, 1, bias=False)

    def forward(self,
                belief_base: torch.Tensor,
                belief_base_sizes: list[int],
                action_tensors: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros_like(belief_base).to('cuda')
        for size in belief_base_sizes:
            mask[:, :size] = 1
        encoded_belief_base = (belief_base * mask).sum(dim=1) / mask.sum(dim=1)

        for block in self.belief_base_encoder:
            encoded_belief_base = block(encoded_belief_base)

        output_repr = torch.cat([encoded_belief_base, action_tensors], dim=-1)
        x = self.hidden(output_repr)
        x = F.relu(x)
        q_values = self.q_value_layer(x)
        return q_values