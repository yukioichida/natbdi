import math

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from sources.fallback_policy.encoder import EncoderModel


class PositionWiseFF(nn.Module):

    def __init__(self, embedding_dim: int, dropout_rate=0.1):
        super().__init__()
        mult = 1
        self.c_fc = nn.Linear(embedding_dim, mult * embedding_dim, bias=False)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(mult * embedding_dim, embedding_dim, bias=False)
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
                 dropout_rate: float = 0.0,
                 n_heads: int = 8):
        super(BeliefTransformerBlock, self).__init__()
        self.scale_attention = scale_attention
        self.attention_dropout = nn.Dropout(dropout_rate)
        self.output_dropout = nn.Dropout(dropout_rate)
        self.layer_norm_1 = nn.LayerNorm(belief_dim, bias=False)
        self.qkv_proj_layer = nn.Linear(belief_dim, belief_dim * 3, bias=False)
        self.mlp = PositionWiseFF(belief_dim, dropout_rate)
        self.layer_norm_2 = nn.LayerNorm(belief_dim, bias=False)
        self.n_heads = n_heads

    def self_attention(self,
                       x: torch.Tensor,
                       belief_base_sizes: list[int]) -> (torch.Tensor, torch.Tensor):
        # TODO: evaluate using flash attention cuda kernels (pytorch implementation -> F.scaled_dot_product)
        q, k, v = self.qkv_proj_layer(x).chunk(3, dim=-1)
        batch_size, seq_len, emb_dim = q.size()
        head_emb_dim = emb_dim // self.n_heads

        # split the emb dim axis, and then transpose for matmul operations
        k = k.view(batch_size, seq_len, self.n_heads, head_emb_dim).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(batch_size, seq_len, self.n_heads, head_emb_dim).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(batch_size, seq_len, self.n_heads, head_emb_dim).transpose(1, 2)  # (B, nh, T, hs)

        attention = torch.matmul(q, k.transpose(-2, -1))
        if self.scale_attention:
            attention = attention / math.sqrt(head_emb_dim)
        pad_mask = self.mask_padding(attention, belief_base_sizes)
        attention[pad_mask] = -1e10
        attention = self.attention_dropout(attention)
        attention = nn.Softmax(dim=-1)(attention)

        y = torch.matmul(attention, v)
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, emb_dim)
        y = self.output_dropout(y)
        return y, attention

    def mask_padding(self, x: torch.Tensor, belief_base_sizes: list[int]) -> torch.Tensor:
        mask = torch.zeros_like(x)
        for i, size in enumerate(belief_base_sizes):
            mask[i, :, :, size:] = 1
        return mask.bool()

    def forward(self, x: torch.Tensor, belief_base_sizes: list[int]) -> (torch.Tensor, torch.Tensor):
        old_x = x
        x = self.layer_norm_1(x)
        x, a = self.self_attention(x, belief_base_sizes)
        x = x + old_x  # residual
        x = self.layer_norm_2(x)
        x = self.mlp(x) + x
        return x, a


class BeliefBaseEncoder(nn.Module):

    def __init__(self, belief_dim: int, n_blocks: int = 10, n_heads: int = 8):
        super(BeliefBaseEncoder, self).__init__()
        self.blocks = nn.ModuleList(
                [BeliefTransformerBlock(belief_dim=belief_dim, n_heads=n_heads) for _ in range(n_blocks)])

    def pooling_belief_base(self, x: torch.Tensor) -> torch.Tensor:
        representation = x[:, 0, :]  # using CLS sounds better in CL learning
        # representation = x.mean(1) # mean pooling looks better in this case
        return representation

    def forward(self, x: torch.Tensor, belief_base_sizes: list[int]) -> (torch.Tensor, torch.Tensor):
        for block in self.blocks:
            x, a = block(x, belief_base_sizes)
        representation = self.pooling_belief_base(x)
        return representation, a


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


class ContrastiveQNetwork(L.LightningModule):

    def __init__(self,
                 belief_dim: int,
                 encoder_model: EncoderModel,
                 n_blocks: int = 2,
                 n_heads: int = 8,
                 cl_temp: float = 0.5):
        super(ContrastiveQNetwork, self).__init__()
        self.belief_base_encoder = BeliefBaseEncoder(belief_dim, n_blocks, n_heads=n_heads)
        self.similarity_function = nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.encoder_model = encoder_model
        self.linear_act = nn.Linear(belief_dim, belief_dim)
        self.linear_belief = nn.Linear(belief_dim, belief_dim)
        self.cl_temp = cl_temp
        self.save_hyperparameters('n_blocks', 'n_heads', 'belief_dim', 'cl_temp')

    def act(self, belief_base, candidate_actions):
        batch = {
                'belief_base': [belief_base],
                'actions': candidate_actions,
                'belief_base_sizes': [len(belief_base)]
        }
        similarity_matrix = self.forward(batch)
        return similarity_matrix

    def _encode_batch(self, batch):
        max_size = max([belief_base_sizes for belief_base_sizes in batch['belief_base_sizes']])
        belief_base_emb = [self.encoder_model.encode_batch(belief_base,
                                                           max_size=max_size,
                                                           include_cls=False)
                           for belief_base in batch['belief_base']]
        belief_base_emb = torch.cat(belief_base_emb, dim=0)
        action_emb = self.encoder_model.encode(batch['actions']).squeeze(0)
        return belief_base_emb, action_emb

    def forward(self, batch):
        belief_base_emb, action_tensor = self._encode_batch(batch)
        belief_base_sizes = batch['belief_base_sizes']
        encoded_belief_base, attention = self.belief_base_encoder(belief_base_emb, belief_base_sizes)
        action_tensor = self.linear_act(action_tensor)
        belief_tensor = self.linear_belief(encoded_belief_base)
        similarity_matrix = self.contrastive_step(belief_tensor, action_tensor)
        return similarity_matrix

    def training_step(self, batch, batch_idx):
        similarity_matrix = self.forward(batch)
        batch_size = similarity_matrix.size(0)  # batch_size, similarity
        cl_label = torch.arange(batch_size, dtype=torch.long).to('cuda')
        loss = F.cross_entropy(similarity_matrix, cl_label)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, batch_size=batch_size)
        return loss

    def contrastive_step(self, belief_base_emb: torch.Tensor, action_emb: torch.Tensor):
        x1 = belief_base_emb
        x2 = action_emb
        similarity_matrix = self.similarity_function(x1.unsqueeze(1), x2.unsqueeze(0)) / self.cl_temp
        return similarity_matrix

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=self.parameters(), lr=5e-5)
        return {"optimizer": optimizer}
