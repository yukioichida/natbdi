import argparse

import lightning
import pandas as pd
import torch
from datasets import Dataset, Features, Sequence, Value
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from scienceworld import ScienceWorldEnv
from torch.utils.data import DataLoader

from sources.fallback_policy.encoder import HFEncoderModel
from sources.fallback_policy.model import ContrastiveQNetwork
from sources.fallback_policy.replay import PrioritizedReplayMemory
from sources.scienceworld.utils import parse_beliefs, parse_goal
from sources.scienceworld.goldpaths import TransitionLoader

TASK_FILES = {
        'task-1-boil': "tabular_task-1-boil.csv"
}


def get_dataset_file(args: argparse.Namespace) -> str:
    dataset_file = TASK_FILES[args.task_name]
    return args.goldpath_dir + dataset_file


def load_argparse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--goldpath_dir", type=str, default="/opt/data/scienceworld-goldpaths/trajectories_csv/")
    parser.add_argument("--task_name", type=str, default='task-1-boil')
    parser.add_argument("--encoder_model", type=str, default='princeton-nlp/sup-simcse-roberta-base')
    parser.add_argument("--epochs", type=int, default=50)
    return parser.parse_args()


def load_replay_buffer(goldpath_file: str) -> PrioritizedReplayMemory:
    transitions = TransitionLoader.load_transitions(goldpath_file, variation=0)  # TODO: remove variation
    replay_memory = PrioritizedReplayMemory(priority_fraction=0.5)
    for transition in transitions:
        replay_memory.push(transition, is_prior=True)
    return replay_memory


def parse_state(observation: str, info: dict, previous_actions: list[dict[str, str | int]], goal: str):
    belief_base = parse_beliefs(observation=obs, look=info['look'], inventory=info['inv']) + [goal]
    for a in previous_action[-2:]:
        belief_base.append(f"You execute {a['action']} at turn {a['turn']}")
    num_beliefs = len(belief_base) + 1  # including CLS
    return belief_base, num_beliefs


if __name__ == '__main__':
    lightning.seed_everything(42)
    args = load_argparse()
    print("Loading encoder model")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder_model = HFEncoderModel(args.encoder_model, device=device)

    checkpoint_file = "../../checkpoints/sup/version_21/epoch=37-step=190-train_loss_epoch=0.803.ckpt"
    model = ContrastiveQNetwork.load_from_checkpoint(checkpoint_file, encoder_model=encoder_model)
    model = model.to(device).train()

    goldpath_file = get_dataset_file(args)
    replay_memory = load_replay_buffer(goldpath_file)

    variation = 0

    env = ScienceWorldEnv()
    env.load("boil", variation, "openDoors")  # TODO: parametrize task name
    goal = parse_goal(env.getTaskDescription())
    goal = f"Your task is to {goal}"
    print(f"Scienceworld environment started. Goal: {goal} - Variation: {variation}")

    for epoch in range(args.epochs):
        max_steps = 36
        action = "look around"
        plan_tracker = []
        previous_actions = []
        acc_reward = 0
        observation, info = env.reset()
        belief_base, num_beliefs = parse_state(observation, info, previous_actions, goal)
        for step in range(max_steps):
            obs, reward, is_done, info = env.step(action)
            if is_done:
                break
            next_belief_base, next_belief_base_size = parse_state(observation=observation,
                                                                  info=info,
                                                                  previous_actions=previous_actions,
                                                                  goal=goal)
            # candidate_actions = available_actions
            next_candidate_actions = info['valid']

            last_belief_base = belief_base

            previous_actions.append({
                    'turn': step,
                    'action': action
            })

            #action = agent.act
