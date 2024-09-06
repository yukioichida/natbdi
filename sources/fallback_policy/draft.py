import json

import lightning as L
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, Features, Sequence, Value
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from scienceworld import ScienceWorldEnv
from torch.utils.data import DataLoader

from sources.fallback_policy.encoder import HFEncoderModel, EncoderModel
from sources.fallback_policy.model import BeliefBaseEncoder
from sources.scienceworld import parse_beliefs

# sys.path.append("..")

L.seed_everything(42)


class ContrastiveQNetwork(L.LightningModule):

    def __init__(self,
                 belief_dim: int,
                 encoder_model: EncoderModel,
                 n_blocks: int = 2):
        super(ContrastiveQNetwork, self).__init__()
        self.belief_base_encoder = BeliefBaseEncoder(belief_dim, n_blocks)
        self.similarity_function = nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.encoder_model = encoder_model
        self.linear_act = nn.Linear(belief_dim, belief_dim)
        self.linear_belief = nn.Linear(belief_dim, belief_dim)

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
        # encoded_belief_base = self.pooling_belief_base(belief_base_emb, belief_base_sizes)

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

    def contrastive_step(self,
                         belief_base_emb: torch.Tensor,
                         action_emb: torch.Tensor):
        # x1 representation (state+action)
        x1 = belief_base_emb
        # x2 representation (goal?)
        x2 = action_emb

        #temp = 0.1 # 0.1 is the best with batch_size = 8
        temp = 0.5 # 0.5 leads to best
        similarity_matrix = self.similarity_function(x1.unsqueeze(1), x2.unsqueeze(0)) / temp
        return similarity_matrix

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=self.parameters(), lr=5e-5)
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
    previous_action = []
    available_actions = []
    for i, trajectory in enumerate(gold_sequence):
        look_around = trajectory['freelook']
        inventory = trajectory['inventory']

        belief_base = parse_beliefs(observation=observation, look=look_around, inventory=inventory)
        belief_base = [b for b in belief_base if len(b) > 0]
        is_done = trajectory['isCompleted']
        if is_done == 'true':
            next_state = ""
            break

        belief_base = belief_base + [goal]

        if trajectory['action'] != 'look around':
            for a in previous_action[-5:]:
                belief_base.append(f"You executed the action {a['action']} at turn {a['turn']}")
            belief_base_sizes = len(belief_base) + 1 if use_cls else len(belief_base)
            action = trajectory['action']
            all_trajectories.append({
                    'belief_base': belief_base,
                    'action': action,
                    'belief_base_sizes': belief_base_sizes,
            })
            previous_action.append({
                    'turn': i,
                    'action': action
            })

            if action not in available_actions:
                available_actions.append(action)

            observation = trajectory['observation']

    return all_trajectories, json_data, available_actions


##
## TRAINING
##

encoder_model = HFEncoderModel("princeton-nlp/sup-simcse-roberta-base", device='cuda')
trajectories, json_data, available_actions = load_goldpaths()

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
dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=8, shuffle=True)
model = ContrastiveQNetwork(768, encoder_model=encoder_model)

base_dir = "cl_step"
tb_logger = TensorBoardLogger(f"logs/{base_dir}")
tb_logger.log_hyperparams(model.hparams)
version = tb_logger.version
filename = base_dir + "/version_" + str(version) + "/" + "v" + str(
        version) + "-{epoch}-{step}-{train_loss_epoch:.3f}"
checkpoint_callback = ModelCheckpoint(dirpath='checkpoints',
                                      monitor='train_loss_epoch',
                                      save_top_k=2,
                                      filename=filename)

trainer = Trainer(max_epochs=40,
                  accelerator='gpu',
                  logger=tb_logger,
                  callbacks=[checkpoint_callback]
                  )
trainer.fit(model, dataloader)


#print(f"BEST CHECKPOINT: {checkpoint_callback.best_model_path}")
#model = ContrastiveQNetwork.load_from_checkpoint(checkpoint_callback.best_model_path,
#                                                 belief_dim=768,
#                                                 encoder_model=encoder_model)
model = model.to('cuda')
model = model.eval()

env = ScienceWorldEnv()
goal = json_data['goldActionSequences'][0]['taskDescription'].split('.')[0]
variation_idx = json_data['goldActionSequences'][0]['variationIdx']

env.load("boil", variation_idx, "openDoors")
with torch.no_grad():
    max_steps = 30
    action = "look around"

    plan = []
    previous_action = []
    for step in range(max_steps):
        obs, reward, is_done, info = env.step(action)

        # if obs is non action that matches the input then
        #    remove action from info['valid']
        # else
        #    do nothing

        print(f" => Step {step} - reward: {reward:.3f} - is_done: {is_done} - action: {action}")
        belief_base = parse_beliefs(observation=obs, look=info['look'], inventory=info['inv']) + [goal]
        belief_base = [b.replace("greenhouse", "green house") for b in belief_base]

        for a in previous_action[-5:]:
            belief_base.append(f"You executed the action {a['action']} at turn {a['turn']}")

        num_beliefs = len(belief_base) + 1 + 1  # including cls
        #candidate_actions = available_actions
        candidate_actions = info['valid']
        #q_values = model.act(belief_base, candidate_actions=info['valid'])
        q_values = model.act(belief_base, candidate_actions=candidate_actions)
        selected_action = q_values.argmax(dim=-1)[0]
        action = candidate_actions[selected_action]
        # if i == 1:
        #   action = "focus on substance in metal pot"
        #print(f"Belief Base: {belief_base}")
        print(f"obs: {obs}")
        print(f"Selected action: {action}")
        values, idxs = torch.sort(q_values.squeeze(0), descending=True)

        top_k = 3
        print(f"\tAction space - Top {top_k}:")
        for i, idx in enumerate(idxs[:top_k]):
            print(f"\t\tCandidate Action: {candidate_actions[idx]} - q_value: {values[i]:.3f}")

        plan.append(action)

        previous_action.append({
                'turn': step,
                'action': action
        })

    print("Plan Executed: ")
    for i, a in enumerate(plan):
        print(f"{i} -  {a}")
