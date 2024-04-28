from typing import NamedTuple


class State(NamedTuple):
    """
    Environment state perceived by the agent
    Means-Ends Reasoning - pag 19
    ... The agent’s current beliefs about the *state* of the environment. ...
    """
    goal: str = ""  # main goal
    beliefs: list[str] = []  # objects that agent are seeing in the current state
    score: float = 0  # reward received by the environment
    error: bool = False
    completed: bool = False
    metadata: dict = {}  # additional information about state


class BeliefBase:

    def __init__(self):
        """
        Agent's Belief base that contains perceived environment state
        """
        self.memory = []

    def belief_update(self, new_state: State):
        """
        Updates the current state of belief base given a new environment state.
        :param new_state: current environment state
        """
        self.memory.append(new_state)

    def get_current_beliefs(self) -> State:
        """
        Retrieve current state stored in the belief base
        :return: List of sentences corresponding to the current state perceived in the belief base
        """
        return self.memory[-1] if len(self.memory) > 0 else []
