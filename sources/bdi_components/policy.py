from typing import Callable

from sources.bdi_components.belief import State
from sources.drrn.drrn_agent import DRRN_Agent
from sources.utils import setup_logger

logger = setup_logger()


class FallbackPolicy:

    def fallback_plan(self,
                      current_state: State,
                      step_function: Callable[[str], State]):
        """
        Selects an action given the current information about the environment.
        :param current_state: current information about the environment
        :param step_function: function that executes an action in textual environment
        :return: environment state after the agent completes the fallback policy execution
        """
        pass


class DRRNFallbackPolicy(FallbackPolicy):

    def __init__(self,
                 drrn_model_file: str,
                 spm_path: str = 'models/spm_models/unigram_8k.model',
                 step_limit: int = 50):
        logger.info(f"Bootstrap DRRN Model with model_file {drrn_model_file}")
        drrn_agent = DRRN_Agent(spm_path=spm_path)
        drrn_agent.load(drrn_model_file)
        self.drrn_agent = drrn_agent
        self.step_limit = step_limit
        self.actions_performed = []
        self.score = 0

    def fallback_plan(self, current_state: State, step_function: Callable[[str], State]):
        info = current_state.metadata
        obs = info['observation']
        for _ in range(self.step_limit):  # stepLimits
            fallback_action = self._select_action(info)
            state = step_function(fallback_action)
            self.actions_performed.append(fallback_action)
            if state.completed:
                break
        self.score = state.score  # score acquired exclusively from DRRN (RL)
        return state

    def _select_action(self, info: dict) -> str:
        """
        Predicts an action and retrieves its string representation
        :param info: environment information
        :return: action selected by the fallback policy, given the current state
        """
        drrn_state = self.drrn_agent.build_state(obs=info['observation'], inv=info['inv'], look=info['look'])
        valid_ids = self.drrn_agent.encode(info['valid'])
        _, action_idx, action_values = self.drrn_agent.act([drrn_state], [valid_ids], sample=False)
        action_idx = action_idx[0]
        action_str = info['valid'][action_idx]
        return action_str
