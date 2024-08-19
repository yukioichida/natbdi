import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from sources.cl_nli.model import SimCSE
from sources.fallback_policy.replay import ReplayMemory, Transition
from sources.scienceworld import parse_beliefs

from sources.fallback_policy.model import SimpleQNetwork, QNetwork

torch.manual_seed(42)

use_transformer = True
if use_transformer:
    network = QNetwork(768, 768, n_blocks=4)
    use_cls = True
else:
    network = SimpleQNetwork(768, 768, 1)
    use_cls = False
network = network.to('cuda')
network.train()

num_parameters = sum(p.numel() for p in network.parameters() if p.requires_grad)
print(f"TRANSFORMER: {use_transformer} - Number of parameters: {num_parameters}")

goldpath_file = "/opt/data/scienceworld-goldpaths/goldsequences-0.json"

with open(goldpath_file) as f:
    json_data = json.load(f)

json_data = json_data['0']

json_data.keys()

ckpt = "/opt/models/simcse_default/version_0/v0-epoch=4-step=18304-val_nli_loss=0.658-train_loss=0.551.ckpt"

model: SimCSE = SimCSE.load_from_checkpoint(ckpt).eval()
hf_model_name = model.hparams['hf_model_name']
tokenizer = AutoTokenizer.from_pretrained(hf_model_name)

print("Creating memory buffer")


def encode_custom(texts: list[str], max_size: int = 25, include_cls: bool = True) -> torch.Tensor:
    if include_cls:
        cls_token = tokenizer.cls_token
        texts = [cls_token] + texts
    pad_size = max_size - len(texts)
    padding = [tokenizer.pad_token for _ in range(pad_size)]
    texts = texts + padding
    tokenized_text = tokenizer(texts, padding='longest', truncation=True,
                               return_tensors='pt').to(model.device)
    embeddings = model.encode(tokenized_text).detach().unsqueeze(0)  # batch axis
    return embeddings


def encode(texts: list[str], max_size: int = 25, include_cls: bool = True) -> torch.Tensor:
    return encode_custom(texts, max_size, include_cls)


gold_sequence = json_data['goldActionSequences'][0]['path']

goal = json_data['goldActionSequences'][0]['taskDescription'].split('.')[0]
variation_idx = json_data['goldActionSequences'][0]['variationIdx']
print(f"Goal: {goal} - variation {variation_idx}")

all_lens = []

trajectories_bdi = []

memory_buffer = ReplayMemory(1000)

last_reward = 0
observation = ""
for i, trajectory in enumerate(gold_sequence):
    look_around = trajectory['freelook']
    inventory = trajectory['inventory']
    belief_base = parse_beliefs(observation=observation, look=look_around, inventory=inventory)

    next_trajectory = gold_sequence[i + 1]
    next_belief_base = parse_beliefs(observation=next_trajectory['observation'],
                                     look=next_trajectory['freelook'],
                                     inventory=next_trajectory['inventory'])

    reward = float(trajectory['score']) - last_reward
    last_reward = float(trajectory['score'])
    is_done = trajectory['isCompleted']
    if is_done == 'true':
        next_state = ""
        print("finish")
        break
        # ou break de repente aqui
    if trajectory['action'] != 'look around':
        memory_buffer.push(
                Transition(
                        belief_base=encode(belief_base + [goal], include_cls=use_cls),
                        num_beliefs=len(belief_base) + 1,  # including goal
                        action=encode([trajectory['action']], max_size=1, include_cls=False),
                        next_belief_base=encode(next_belief_base + [goal]),
                        num_next_beliefs=len(next_belief_base) + 1,
                        next_action=encode([next_trajectory['action']], max_size=1, include_cls=False),
                        reward=reward,
                        done=True if is_done == 'true' else False
                )
        )
    observation = trajectory['observation']

print(" ============ Training q-network ============ ")
# q-learning
GAMMA = 0.99
BATCH_SIZE = 16

optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)


def contrastive_loss(belief_base_emb, num_belief_emb, action_emb):
    # USING CONTRASTIVE LOSS  (cross entropy) does not work well
    # maybe because we did not need to use a softmax function at the top of network to predict q-values
    batch_size, _, _ = belief_base_emb.size()

    belief_q_values = []
    for belief_idx in range(BATCH_SIZE):
        c_belief_base = belief_base_emb[belief_idx, :, :].unsqueeze(0)
        c_belief_base = c_belief_base.repeat(batch_size, 1, 1)
        q_values = network(belief_base=c_belief_base,
                           belief_base_sizes=[num_belief_emb[belief_idx] for _ in range(batch_size)],
                           action_tensors=action_emb)
        belief_q_values.append(q_values.squeeze(0))

    all_q_values = torch.cat(belief_q_values, dim=-1)
    labels = torch.zeros_like(all_q_values).to('cuda')
    labels = labels.fill_diagonal_(1)
    all_q_values = all_q_values.view(batch_size * batch_size, 1)
    labels = labels.view(batch_size * batch_size, 1)
    # return F.mse_loss(all_q_values, labels)
    return F.smooth_l1_loss(all_q_values, labels)


for epoch in range(100):
    batch = memory_buffer.sample(BATCH_SIZE)
    belief_base_emb = torch.cat([b.belief_base for b in batch], dim=0)
    batch_size, _, belief_dim = belief_base_emb.size()
    num_belief_emb = [b.num_beliefs for b in batch]
    actions = torch.cat([b.action for b in batch]).squeeze(1)  # removing mid axis [bs, ?, a_dim] -> [bs, a_dim]

    optimizer.zero_grad()
    loss = contrastive_loss(belief_base_emb, num_belief_emb, actions)
    if epoch % 10 == 0:
        print(f"epoch {epoch} - loss {loss.item(): .4f}")
    loss.backward()
    optimizer.step()
    # break
    # nn.utils.clip_grad_norm_(network.parameters(), 1.)

print(f"epoch {epoch} - loss {loss.item(): .4f}")

network = network.eval()
with torch.no_grad():
    # print(f"attention {attn}")
    # simple network epoch 299 - loss  0.0219
    print("Test in a single trajectory")
    turn = 8
    expected_action = gold_sequence[turn]['action']
    annotated_belief_base = parse_beliefs(observation=gold_sequence[turn - 1]['observation'],
                                          look=gold_sequence[turn]['freelook'],
                                          inventory=gold_sequence[turn]['inventory']) + [
                                    goal]  # ['your task is to melt gallium']
    # annotated_belief_base = ['This room is called the kitchen', 'You see a anchor', 'you see a metal pot'] + [goal]
    print(annotated_belief_base)

    encoded_belief_base = encode(annotated_belief_base, max_size=len(annotated_belief_base) + 1, include_cls=use_cls)

    # candidate_actions = info['valid']
    candidate_actions = ['focus on bedroom door', 'open door to kitchen', 'go to kitchen', 'deactivate sink'] + [
            expected_action]
    encoded_actions = encode(candidate_actions, include_cls=False, max_size=len(candidate_actions))
    encoded_actions = encoded_actions.squeeze(0)
    num_actions, action_dim = encoded_actions.size()
    repeat_encoded_belief_base = encoded_belief_base.repeat(num_actions, 1, 1)
    q_values = network(belief_base=repeat_encoded_belief_base,
                       belief_base_sizes=[len(annotated_belief_base) + 1 for _ in range(num_actions)],
                       action_tensors=encoded_actions)

    print(f"Expected action: {expected_action}")
    values, idxs = torch.sort(q_values.squeeze(1), descending=True)
    for i, idx in enumerate(idxs[:5]):
        print(f"act {candidate_actions[idx]} - q_value: {values[i]:.3f}")
