import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from sources.cl_nli.model import SimCSE
from sources.fallback_policy.model import QNetwork
from sources.scienceworld import parse_beliefs

from sources.fallback_policy.replay import ReplayMemory, Transition

ckpt = "/opt/models/simcse_default/version_0/v0-epoch=4-step=18304-val_nli_loss=0.658-train_loss=0.551.ckpt"

model: SimCSE = SimCSE.load_from_checkpoint(ckpt).eval()
hf_model_name = model.hparams['hf_model_name']
tokenizer = AutoTokenizer.from_pretrained(hf_model_name)

def encode(texts: list[str], max_size: int = 25, is_action: bool = False) -> torch.Tensor:
    pad_size = max_size - len(texts)
    padding = [tokenizer.pad_token for _ in range(pad_size - 1)]
    cls_token = tokenizer.cls_token
    if not is_action:
        texts = [cls_token] + texts + padding
    tokenized_text = tokenizer(texts, padding='longest', truncation=True,
                               return_tensors='pt').to(model.device)
    embeddings = model.encode(tokenized_text).detach().unsqueeze(0)  # batch axis
    return embeddings

embeddings_a = encode(['you see a door, which is open', 'you see a door, which is closed'], max_size=3)
emb_a = embeddings_a[0, 1, :]
emb_b = embeddings_a[0, 2, :]

network = QNetwork(768, 768, 768, n_blocks=5)
network = network.to('cuda')

enc_a = network.belief_base_encoder(embeddings_a, [3])

belief_test = ['you see a door, which is open', 'you see a door, which is not closed']
print(len(belief_test))
embeddings_b = encode(belief_test, max_size=3)

enc_b = network.belief_base_encoder(embeddings_b, [3])

sim = nn.CosineSimilarity(dim=-1)(enc_a, enc_b)
print(sim)