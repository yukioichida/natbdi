from sources.bdi_components.policy import FallbackPolicy
from sources.fallback_policy.replay import ReplayMemory, Transition


class FallbackBDIPolicy(FallbackPolicy):


    def __init__(self):
        self.replay_memory = ReplayMemory(10000)

    def train(self, transition: Transition):
        pass

