import argparse

import time

import lightning
import pandas as pd
import torch
from datasets import Dataset, Features, Sequence, Value
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from scienceworld import ScienceWorldEnv
from torch.utils.data import DataLoader

from sources.cl_nli.model import SimCSE
from sources.fallback_policy.encoder import HFEncoderModel, CustomSimCSEModel
from sources.fallback_policy.model import ContrastiveQNetwork
from sources.scienceworld.utils import parse_beliefs, parse_goal

TASK_FILES = {
        'task-1-boil': "tabular_task-1-boil.csv"
}


def change_action_string(action: str):
    DICT = {
            'use thermometer in inventory on substance in metal pot': 'use thermometer on substance in metal pot',
            'examine substance in metal pot': 'look at substance in metal pot'
    }
    if action in DICT:
        return DICT[action]
    else:
        return action

def load_goldpath_dataset(goldpath_file: str, variation: int = None):
    goldpath_df = pd.read_csv(goldpath_file)
    if variation is not None:
        goldpath_df = goldpath_df[goldpath_df['variation_idx'] == variation]
    goldpath_df = goldpath_df.sort_values("turn")

    all_trajectories = []
    previous_actions = []
    observation = ""
    for i, row in goldpath_df.iterrows():
        belief_base = parse_beliefs(observation=observation, look=row['look_around'], inventory=row['inventory'])
        belief_base = [b for b in belief_base if len(b) > 0] + [row['goal']]
        for a in previous_actions[-2:]:
            belief_base.append(f"You execute {a['action']} at turn {a['turn']}")

        belief_base_sizes = len(belief_base) + 1
        action = row['action']
        #action = change_action_string(action)
        all_trajectories.append({
                'belief_base': belief_base,
                'action': action,
                'belief_base_sizes': belief_base_sizes,
        })

        previous_actions.append({
                'turn': row['turn'],
                'action': action
        })

        observation = row['observation']

    trajectories_pd = pd.DataFrame(all_trajectories)
    dataset = Dataset.from_pandas(trajectories_pd, features=Features({
            "belief_base": Sequence(Value(dtype="string")),
            "action": Value(dtype="string"),
            "belief_base_sizes": Value(dtype="int32")
    }))
    return dataset


def get_dataset_file(args: argparse.Namespace) -> str:
    dataset_file = TASK_FILES[args.task_name]
    return args.goldpath_dir + dataset_file


def collate_fn(data):
    actions = [d['action'] for d in data]
    belief_base_sizes = [d['belief_base_sizes'] for d in data]
    belief_base = [d['belief_base'] for d in data]

    return {'actions': actions,
            'belief_base_sizes': belief_base_sizes,
            'belief_base': belief_base}


def load_argparse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--goldpath_dir", type=str, default="/opt/data/scienceworld-goldpaths/trajectories_csv/")
    parser.add_argument("--task_name", type=str, default='task-1-boil')
    parser.add_argument("--encoder_model", type=str, default='princeton-nlp/sup-simcse-roberta-base')
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_blocks", type=int, default=2)
    parser.add_argument("--cl_temp", type=float, default=0.5)
    parser.add_argument("--mean_pooling", action='store_true', default=False)
    parser.add_argument("--n_heads", type=int, default=8)
    return parser.parse_args()


if __name__ == '__main__':

    start_time = time.time()
    lightning.seed_everything(42)
    args = load_argparse()
    print(args)
    print("Loading encoder model")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    custom_encoder = False
    print(f"Custom encoder model {custom_encoder}")
    if custom_encoder:
        simcse = SimCSE.load_from_checkpoint(
            '/opt/models/simcse_default/version_0/v0-epoch=4-step=18304-val_nli_loss=0.658-train_loss=0.551.ckpt')

        encoder_model = CustomSimCSEModel(simcse)
    else:
        encoder_model = HFEncoderModel(args.encoder_model, device=device)

    goldpath_file = get_dataset_file(args)
    #dataset = load_goldpath_dataset(goldpath_file, variation=0)  # TODO: remove variation
    dataset = load_goldpath_dataset(goldpath_file)
    # gerar os embeddings aqui, talvez criar um huggingface datasets
    dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=args.batch_size, shuffle=True)

    embedding_dim = encoder_model.embedding_dim
    model = ContrastiveQNetwork(embedding_dim,
                                encoder_model=encoder_model,
                                n_blocks=args.n_blocks,
                                n_heads=args.n_heads,
                                cl_temp=args.cl_temp,
                                mean_pooling=args.mean_pooling)

    base_dir = "sup_all"
    tb_logger = TensorBoardLogger(f"logs/{base_dir}")
    tb_logger.log_hyperparams(model.hparams)
    tb_logger.log_hyperparams(args)
    tb_logger.log_hyperparams({'custom_encoder': custom_encoder})
    filename = base_dir + "/version_" + str(tb_logger.version) + "/{epoch}-{step}-{train_loss_epoch:.3f}"

    checkpoint_callback = ModelCheckpoint(dirpath='checkpoints',
                                          monitor='train_loss_epoch',
                                          save_top_k=2,
                                          filename=filename)

    trainer = Trainer(max_epochs=args.epochs,
                      accelerator='gpu',
                      logger=tb_logger,
                      callbacks=[checkpoint_callback])
    trainer.fit(model, dataloader)

    end_time =  time.time() - start_time
    print(f"TRAINING DURATION --- {end_time}")

    # model = ContrastiveQNetwork.load_from_checkpoint("checkpoints/sup/version_3/epoch=37-step=190-train_loss_epoch=0.812.ckpt", encoder_model = encoder_model)
    #eval_in_env(model, variation=0)
