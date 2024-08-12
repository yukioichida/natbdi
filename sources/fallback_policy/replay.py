import numpy as np
from collections import namedtuple

Transition = namedtuple(typename='Transition',
                        field_names=('state', 'action', 'reward', 'done', 'next_state', 'next_action', 'done'))


class ReplayMemory:

    def __init__(self, capacity:int, seed: int = 42):
        self.capacity = capacity
        self.memory = []
        self.current_position = 0
        self.rand = np.random.RandomState(seed)

    def push(self, *transition_args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.current_position] = Transition(*transition_args)
        self.current_position = (self.current_position + 1) % self.capacity

    def __len__(self):
        return len(self.memory)

    def sample(self, batch_size:int):
        idxs = []
        if len(self.memory) > batch_size:
            idxs = self.rand.choice(range(len(self.memory)), batch_size)
        else:
            print("Wait accumulating more transitions") # TODO: use loggers
        sampled_memories = [self.memory[i] for i in idxs]
        return sampled_memories

