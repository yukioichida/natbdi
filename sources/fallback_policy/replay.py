import numpy as np
from collections import namedtuple

Transition = namedtuple(typename='Transition',
                        field_names=('belief_base',
                                     'belief_base_size',
                                     'action',
                                     'reward',
                                     'next_belief_base',
                                     'next_belief_base_size',
                                     'next_action',
                                     'done'))


class ReplayMemory:

    def __init__(self, capacity: int, seed: int = 42):
        self.capacity = capacity
        self.memory = []
        self.current_position = 0
        self.rand = np.random.RandomState(seed)

    def push(self, transition: Transition):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.current_position] = transition
        self.current_position = (self.current_position + 1) % self.capacity

    def __len__(self):
        return len(self.memory)

    def sample(self, batch_size: int):
        idxs = []
        if len(self.memory) > batch_size:
            idxs = self.rand.choice(range(len(self.memory)), batch_size)
        else:
            print("Wait accumulating more transitions")  # TODO: use loggers
        sampled_memories = [self.memory[i] for i in idxs]
        return sampled_memories


class PrioritizedReplayMemory(object):

    def __init__(self, total_capacity=100000, priority_fraction=0.0, seed=42):
        # Stored
        self.total_capacity = total_capacity
        self.priority_fraction = priority_fraction
        self.seed = seed

        # Calculated at init
        self.alpha_capacity = int(total_capacity * priority_fraction)
        self.beta_capacity = total_capacity - self.alpha_capacity

        # Partitioned memory
        self.alpha_memory, self.beta_memory = [], []
        self.alpha_position, self.beta_position = 0, 0

        self.random_state = np.random.RandomState(seed)

    def push(self, transition: Transition, is_prior=False):
        if is_prior:
            if len(self.alpha_memory) < self.alpha_capacity:
                self.alpha_memory.append(None)
            self.alpha_memory[self.alpha_position] = transition
            self.alpha_position = (self.alpha_position + 1) % self.alpha_capacity
        else:
            if len(self.beta_memory) < self.beta_capacity:
                self.beta_memory.append(None)
            self.beta_memory[self.beta_position] = transition
            self.beta_position = (self.beta_position + 1) % self.beta_capacity

    def _get_random_transitions(self, memory: list, sample_size: int) -> list[int]:
        selected_idx = self.random_state.choice(range(len(memory)), sample_size)
        return memory[selected_idx]

    def sample(self, batch_size):
        from_alpha = min(int(self.priority_fraction * batch_size), len(self.alpha_memory))
        alpha_memories = self._get_random_transitions(memory=self.alpha_memory, sample_size=from_alpha)
        from_beta = min(batch_size - int(self.priority_fraction * batch_size), len(self.beta_memory))
        beta_memories = self._get_random_transitions(memory=self.beta_memory, sample_size=from_beta)
        result = alpha_memories + beta_memories
        self.random_state.shuffle(result)
        return result

    def __len__(self):
        return len(self.alpha_memory) + len(self.beta_memory)
