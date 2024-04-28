import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from sources.bdi_components.rl.model import TransformerBDI

class FallbackPolicy(L.LightningModule):

    def __init__(self, model_dim:int):
        super().__init__()
        self.belief_encoder = TransformerBDI(model_dim=model_dim, n_heads=1)
        # receives the embedding and predicts 1 (select action) or 0 (ignore action)
        self.output_layer = nn.Linear(model_dim, 1)

    def forward(self, input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        encoded_state = self.belief_encoder(x=input, mask=mask)
        return encoded_state

    def training_step(self, batch, batch_idx):
        input, mask, action, y = batch
        state = self.belief_encoder(x=input, mask=mask)

        # joins action with state to generate the prediction
        h = torch.matmul(state, action)
        y_hat = self.output_layer(h)

        loss = F.binary_cross_entropy(y_hat, target=y)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())








