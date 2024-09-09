import json

import lightning as L
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, Features, Sequence, Value
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from scienceworld import ScienceWorldEnv
from torch.utils.data import DataLoader

from sources.fallback_policy.encoder import HFEncoderModel, EncoderModel
from sources.fallback_policy.model import BeliefBaseEncoder, ContrastiveQNetwork
from sources.scienceworld import parse_beliefs

# sys.path.append("..")

L.seed_everything(42)


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

# print(f"BEST CHECKPOINT: {checkpoint_callback.best_model_path}")
# model = ContrastiveQNetwork.load_from_checkpoint(checkpoint_callback.best_model_path,
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
        # candidate_actions = available_actions
        candidate_actions = info['valid']
        # q_values = model.act(belief_base, candidate_actions=info['valid'])
        q_values = model.act(belief_base, candidate_actions=candidate_actions)
        selected_action = q_values.argmax(dim=-1)[0]
        action = candidate_actions[selected_action]
        # if i == 1:
        #   action = "focus on substance in metal pot"
        # print(f"Belief Base: {belief_base}")
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
