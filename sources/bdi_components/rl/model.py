import math
import torch

import torch.nn as nn


class MultiHeadAttention(nn.Module):

    def __init__(self, n_heads: int, model_dim: int):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.model_dim = model_dim
        self.head_dim = model_dim // n_heads

    def scaled_dot_product_attention(self,
                                     q: torch.Tensor,
                                     k: torch.Tensor,
                                     v: torch.Tensor,
                                     mask: torch.Tensor = None,
                                     inf_value: float = 1e-9) -> (torch.Tensor, torch.Tensor):
        # Calculate attention scores (allignment between inputs)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        # remove masked (PAD) beliefs from allignment computation
        attn_scores = attn_scores.masked_fill(mask == 0, inf_value)
        # Softmax is applied to obtain attention probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)
        # Multiply by values to obtain the final output
        output = torch.matmul(attn_probs, v)
        return output, attn_probs

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, model_dim = x.size()
        # segment each belief dim into n_heads -> [b1, b2, b3] = [[b1_h1, b1_hn], [b2_h1, b2_hn], [b3_h1, 3_hn]]
        splitted_input = x.view(batch_size, seq_len, self.n_heads, self.head_dim)
        return splitted_input.transpose(1, 2)  # [bs, seq_len, h_heads, head_dim] -> [bs, h_heads, seq_len, head_dim]

    def _join_heads(self, x: torch.Tensor) -> torch.Tensor:
        # Combine the multiple heads back to original shape
        batch_size, _, seq_len, head_dim = x.size()
        # [bs, n_heads, seq_len, head_dim] -> [bs, seq_len, n_heads, head_dim] ->  [bs, seq_len, n_heads * head_dim]
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                mask: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        # Apply linear transformations and split heads
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # Perform scaled dot-product attention
        attn_output, attn_probs = self.scaled_dot_product_attention(q, k, v, mask=mask)

        # Combine heads and apply output transformation
        output = self.combine_heads(attn_output)
        return output, attn_probs


class TransformerBDI(nn.Module):

    def __init__(self, model_dim: int, pooling_pos: int, n_heads: int):
        """
        :param model_dim: Dimension of each encoded belief
        :param pooling_pos: Position to be pooled in the input vector
        :param n_heads: Number of heads in multi-head attention
        """
        super(TransformerBDI, self).__init__()
        self.pooling_pos = pooling_pos
        self.n_heads = n_heads
        self.transformer_attention = MultiHeadAttention(model_dim=model_dim, n_heads=n_heads)

    def forward(self, x: torch.Tensor, mask_pos: torch.Tensor) -> torch.Tensor:
        q, k, v = x, x, x

        # TODO: using linear transformation in q,k e v (q_w, k_w, v_w)
        attn_output, attn_probs = self.transformer_attention(q, k, v, mask=mask_pos)
        # TODO: explore the attention of goal vector with all other representation
        # TODO: evaluate the use of positionwise-ff (q_w, k_w, v_w), dropout and LayerNorm in the output

        # extract the goal latent representation
        pooled_vector = attn_output[:, self.pooling_pos, :]
        return pooled_vector  # [batch_size, model_dim] -> output goal vector



