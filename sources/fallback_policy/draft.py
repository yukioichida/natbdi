import json
import pandas as pd

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import Dataset, Features, Sequence, Value
from lightning import Trainer

from sources.fallback_policy.encoder import HFEncoderModel, EncoderModel
from sources.fallback_policy.model import BeliefBaseEncoder
from sources.scienceworld import parse_beliefs


from torch.utils.data import DataLoader


class ContrastiveQNetwork(L.LightningModule):

    def __init__(self,
                 belief_dim: int,
                 encoder_model: EncoderModel,
                 n_blocks: int = 4):
        super(ContrastiveQNetwork, self).__init__()
        self.belief_base_encoder = BeliefBaseEncoder(belief_dim, n_blocks)
        self.similarity_function = nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.encoder_model = encoder_model
        self.linear_act = nn.Linear(belief_dim, belief_dim)
        self.linear_belief = nn.Linear(belief_dim, belief_dim)


    def _encode_batch(self, batch):

        max_size = max([len(belief_base) for belief_base in batch['belief_base']])
        belief_base_emb = [self.encoder_model.encode_batch(belief_base,
                                                           max_size=max_size,
                                                           include_cls=False)
                           for belief_base in batch['belief_base']]
        belief_base_emb = torch.cat(belief_base_emb, dim=0)
        action_emb = self.encoder_model.encode(batch['actions']).squeeze(0)
        return belief_base_emb, action_emb

    def pooling_belief_base(self, belief_base, belief_base_sizes):
        batch_belief_base = []
        for batch_idx, size in enumerate(belief_base_sizes):
            mean_belief_base = belief_base[batch_idx, :size, :].mean(dim=0).unsqueeze(0)
            batch_belief_base.append(mean_belief_base)

        encoded_belief_base = torch.cat(batch_belief_base, dim=0)
        return encoded_belief_base

    def forward(self, batch):
        belief_base_emb, action_tensor = self._encode_batch(batch)
        belief_base_sizes = batch['belief_base_sizes']
        encoded_belief_base, attention = self.belief_base_encoder(belief_base_emb, belief_base_sizes)
        #encoded_belief_base = self.pooling_belief_base(belief_base_emb, belief_base_sizes)

        action_tensor = self.linear_act(action_tensor)
        belief_tensor = self.linear_belief(encoded_belief_base)
        similarity_matrix = self.contrastive_step(belief_tensor, action_tensor)
        return similarity_matrix

    def training_step(self, batch, batch_idx):
        similarity_matrix = self.forward(batch)
        batch_size = similarity_matrix.size(0)  # batch_size, similarity
        cl_label = torch.arange(batch_size, dtype=torch.long).to('cuda')
        loss = F.cross_entropy(similarity_matrix, cl_label)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size)
        return loss

    def contrastive_step(self,
                         belief_base_emb: torch.Tensor,
                         action_emb: torch.Tensor):
        # x1 representation (state+action)
        x1 = belief_base_emb
        # x2 representation (goal?)
        x2 = action_emb

        similarity_matrix = self.similarity_function(x1.unsqueeze(1), x2.unsqueeze(0)) / 0.1
        return similarity_matrix

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=self.parameters(), lr=1e-4)
        # scheduler = get_linear_schedule_with_warmup(optimizer, num_training_steps=self.hparams['training_epochs'], num_warmup_steps=0)
        # return {"optimizer": optimizer, "lr_scheduler": scheduler}

        return {"optimizer": optimizer}


def load_goldpaths():
    goldpath_file = "/opt/data/scienceworld-goldpaths/goldsequences-0.json"
    with open(goldpath_file) as f:
        data = json.load(f)
    json_data = data['0']

    gold_sequence = json_data['goldActionSequences'][0]['path']
    goal = json_data['goldActionSequences'][0]['taskDescription'].split('.')[0]
    variation_idx = json_data['goldActionSequences'][0]['variationIdx']
    print(f"Goal: {goal} - variation {variation_idx}")

    last_reward = 0
    use_cls = True
    observation = ""
    all_trajectories = []
    for i, trajectory in enumerate(gold_sequence):
        look_around = trajectory['freelook']
        inventory = trajectory['inventory']
        belief_base = parse_beliefs(observation=observation, look=look_around, inventory=inventory)
        is_done = trajectory['isCompleted']
        if is_done == 'true':
            next_state = ""
            break

        belief_base = belief_base + [goal]
        belief_base_sizes = len(belief_base) + 1 if use_cls else len(belief_base)
        if trajectory['action'] != 'look around':
            all_trajectories.append({
                    'belief_base': belief_base,
                    'action': trajectory['action'],
                    'belief_base_sizes': belief_base_sizes,
            })

    return all_trajectories


encoder_model = HFEncoderModel("princeton-nlp/sup-simcse-roberta-base", device='cuda')
trajectories = load_goldpaths()

trajectories_pd = pd.DataFrame(trajectories)
dataset = Dataset.from_pandas(trajectories_pd, features=Features({
        "belief_base": Sequence(Value(dtype="string")),
        "action": Value(dtype="string"),
        "belief_base_sizes": Value(dtype="int32")
}))


def collate_fn(data):
    # tem que fazer o encode aqui, para entregar batchs de vetores prontos
    actions = [d['action'] for d in data]
    belief_base_sizes = [d['belief_base_sizes'] for d in data]
    belief_base = [d['belief_base'] for d in data]

    return {'actions': actions,
            'belief_base_sizes': belief_base_sizes,
            'belief_base': belief_base}

print(len(dataset))
dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=16, shuffle=True)
model = ContrastiveQNetwork(768, encoder_model=encoder_model)

trainer = Trainer(max_epochs=40,
                  accelerator='gpu')
trainer.fit(model, dataloader)