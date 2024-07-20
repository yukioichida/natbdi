import unittest

import torch
from scienceworld import ScienceWorldEnv
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from sources.agent import NatBDIAgent
from sources.bdi_components.belief import State
from sources.bdi_components.inference import NLIModel
from sources.bdi_components.plans import PlanLibrary
from sources.bdi_components.policy import DRRNFallbackPolicy
from sources.scienceworld import parse_state, load_step_function

import sys


def load_plan_library(plan_file: str):
    """
    Loads the natural language plan library
    :param plan_file: file containing plans written in natural language
    :return: plan library
    """
    pl = PlanLibrary()
    pl.load_plans_from_file(plan_file)
    pl.load_plans_from_file("../plans/plans_nl/plan_common.plan")
    pl.load_plans_from_file("../plans/plans_nl/plans_navigation.txt")
    return pl


def run_agent(plan_library: PlanLibrary,
              nli_model: NLIModel,
              env: ScienceWorldEnv,
              drrn_model_file: str) -> (State, NatBDIAgent):
    """
    Executes the experiment phase where the BDI agent reasons over the environment state and call plans.
    :param plan_library: Plan Library containing plans
    :param nli_model: Natural Language Inference model
    :param env: scienceworld environment
    :param drrn_model_file: File with DRRN weights trained
    :return: Last state achieved by the BDI agent with its own instance.
    """

    fallback_policy = DRRNFallbackPolicy(drrn_model_file, spm_path="../models/spm_models/unigram_8k.model")
    main_goal = env.getTaskDescription() \
        .replace(". First, focus on the thing. Then,", "") \
        .replace(". First, focus on the substance. Then, take actions that will cause it to change its state of matter",
                 "") \
        .replace("move", "by moving") \
        .replace("Your task is to", "") \
        .replace(".", "").strip()

    env.reset()
    step_function = load_step_function(env, main_goal)

    # initial state
    observation, reward, isCompleted, info = env.step('look around')
    current_state = parse_state(observation=observation,
                                info=info,
                                task=main_goal)
    bdi_agent = NatBDIAgent(plan_library=plan_library, nli_model=nli_model, fallback_policy=fallback_policy)
    bdi_agent.act(state=current_state, step_function=step_function, goal=main_goal)
    return bdi_agent


def init_nli_model(nli_model: str, device: str) -> NLIModel:
    model = AutoModelForSequenceClassification \
        .from_pretrained(nli_model, torch_dtype=torch.float16) \
        .to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(nli_model)
    nli_model = NLIModel(model=model, tokenizer=tokenizer, device=device)
    return nli_model


class NatBDIAgentTest(unittest.TestCase):

    def setUp(self) -> None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        hf_nli_model = 'roberta-large-mnli'
        self.nli_model = init_nli_model(hf_nli_model, device)
        # loading scienceworld env
        self.env = ScienceWorldEnv("", "", envStepLimit=100)

    def test_run_melt_all_plans(self):
        task = "melt"
        self.env.load(task, 26, simplificationStr="easy")

        drrn_model_file = "test_data/model-steps80000-eps614.pt"

        plan_file = f"test_data/plan_{task}_100.plan"
        pl = load_plan_library(plan_file)
        self.nli_model.reset_statistics()
        agent = run_agent(plan_library=pl,
                          nli_model=self.nli_model,
                          env=self.env,
                          drrn_model_file=drrn_model_file)
        last_state = agent.belief_base.get_current_beliefs()
        self.assertEqual(last_state.completed, True)

    def test_run_melt_no_plans(self):
        task = "melt"
        self.env.load(task, 26, simplificationStr="easy")

        drrn_model_file = "test_data/model-steps80000-eps614.pt"

        plan_file = f"test_data/plan_{task}_0.plan"
        pl = load_plan_library(plan_file)
        self.nli_model.reset_statistics()
        agent = run_agent(plan_library=pl,
                          nli_model=self.nli_model,
                          env=self.env,
                          drrn_model_file=drrn_model_file)
        last_state = agent.belief_base.get_current_beliefs()
        self.assertEqual(last_state.completed, False)


if __name__ == '__main__':
    sys.path.insert(0, '.')
    NatBDIAgentTest.main()
