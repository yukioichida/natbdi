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
from sources.fallback_policy.replay import Transition
from sources.scienceworld.utils import parse_beliefs, parse_goal


class TransitionLoader:

    @staticmethod
    def load_transitions(goldpath_file: str, variation: int) -> list:
        goldpath_df = pd.read_csv(goldpath_file)
        goldpath_df = goldpath_df[goldpath_df['variation_idx'] == variation].sort_values("turn")

        all_paths = []
        previous_actions = []
        observation = ""
        for i, row in goldpath_df.iterrows():
            belief_base = parse_beliefs(observation=observation, look=row['look_around'], inventory=row['inventory'])
            belief_base = [b for b in belief_base if len(b) > 0] + [row['goal']]
            for a in previous_actions[-2:]:
                belief_base.append(f"You execute {a['action']} at turn {a['turn']}")

            belief_base_size = len(belief_base) + 1
            action = row['action']
            # action = change_action_string(action)
            all_paths.append({'belief_base': belief_base,
                              'action': action,
                              'belief_base_size': belief_base_size,
                              'reward': row['reward'],
                              'done': row['done']})
            previous_actions.append({'turn': row['turn'], 'action': action})

            observation = row['observation']

        # Create transitions from steps
        all_transitions = []
        for i, path in enumerate(all_paths):
            next_idx = i + 1
            if next_idx < len(all_paths):
                next_trajectory = all_paths[next_idx]
                next_path = {
                        'next_belief_base': path['belief_base'],
                        'next_belief_base_size': path['belief_base_size'],
                        'next_action': path['action']
                }
            else:
                next_path = {}

            transition_info = path | next_path
            all_transitions.append(Transition(**transition_info))
        return all_transitions
