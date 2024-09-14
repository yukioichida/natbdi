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

    def __init__(self,
                 belief_dim: int,
                 n_blocks: int = 10,
                 n_heads: int = 8,
                 mean_pooling: bool = False):
        super(BeliefBaseEncoder, self).__init__()
        self.blocks = nn.ModuleList(
                [BeliefTransformerBlock(belief_dim=belief_dim, n_heads=n_heads) for _ in range(n_blocks)])
        self.mean_pooling = mean_pooling

    def pooling_belief_base(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean_pooling:
            return x.mean(dim=1)
        else:
            return x[:, 0, :]  # CLS

    def forward(self, x: torch.Tensor, belief_base_sizes: list[int]) -> (torch.Tensor, torch.Tensor):
        for block in self.blocks:
            x, a = block(x, belief_base_sizes)
        representation = self.pooling_belief_base(x)
        return representation, a


class ContrastiveQNetwork(L.LightningModule):

    def __init__(self,
                 belief_dim: int,
                 encoder_model: EncoderModel,
                 proj_dim: int = 64,
                 n_blocks: int = 2,
                 n_heads: int = 8,
                 cl_temp: float = 0.5,
                 mean_pooling: bool = False,
                 lr: float = 5e-5):
        super(ContrastiveQNetwork, self).__init__()
        self.belief_base_encoder = BeliefBaseEncoder(belief_dim,
                                                     n_blocks=n_blocks,
                                                     n_heads=n_heads,
                                                     mean_pooling=mean_pooling)
        self.similarity_function = nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.encoder_model = encoder_model
        self.linear_act = nn.Sequential(
                nn.Linear(belief_dim, belief_dim),
                nn.ReLU(),
                nn.Linear(belief_dim, proj_dim)
        )
        self.linear_belief = nn.Sequential(
                nn.Linear(belief_dim, belief_dim),
                nn.ReLU(),
                nn.Linear(belief_dim, proj_dim)
        )
        self.save_hyperparameters('n_blocks', 'n_heads', 'belief_dim', 'cl_temp', 'proj_dim', 'mean_pooling', 'lr')

    def act(self, belief_base, candidate_actions):
        batch = {
                'belief_base': [belief_base],
                'actions': candidate_actions,
                'belief_base_sizes': [len(belief_base) + 1]
        }
        similarity_matrix = self.forward(batch)
        return similarity_matrix

    def _encode_batch(self, batch):
        use_cls = not self.hparams['mean_pooling']
        max_size = max([belief_base_sizes for belief_base_sizes in batch['belief_base_sizes']])
        belief_base_emb = [self.encoder_model.encode_batch(belief_base,
                                                           max_size=max_size,
                                                           include_cls=use_cls)
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
        similarity_matrix = self.similarity_function(x1.unsqueeze(1), x2.unsqueeze(0)) / self.hparams['cl_temp']
        return similarity_matrix

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=self.parameters(), lr=self.hparams['lr'])
        return {"optimizer": optimizer}
