import argparse
import os.path
import random
import json
import re
from os import listdir
from os.path import isfile, join

import pandas as pd
import torch
from scienceworld import ScienceWorldEnv
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from sources.agent import NatBDIAgent
from sources.bdi_components.belief import State
from sources.bdi_components.inference import NLIModel
from sources.bdi_components.plans import PlanLibrary
from sources.bdi_components.policy import DRRNFallbackPolicy
from sources.scienceworld import parse_state, load_step_function
from sources.utils import setup_logger

logger = setup_logger()


def load_plan_library(plan_file: str):
    """
    Loads the natural language plan library
    :param plan_file: file containing plans written in natural language
    :return: plan library
    """
    pl = PlanLibrary()
    pl.load_plans_from_file(plan_file)
    pl.load_plans_from_file("plans/plans_nl/plan_common.plan")
    pl.load_plans_from_file("plans/plans_nl/plans_navigation.txt")
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

    logger.info(f"Start BDI reasoning phase")
    fallback_policy = DRRNFallbackPolicy(drrn_model_file)
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


def init_nli_model(args) -> NLIModel:
    model = AutoModelForSequenceClassification \
        .from_pretrained(args.nli_model, torch_dtype=torch.float16) \
        .to(args.device).eval().float()
    tokenizer = AutoTokenizer.from_pretrained(args.nli_model)
    nli_model = NLIModel(model=model, tokenizer=tokenizer, device=args.device)
    return nli_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='melt')
    parser.add_argument('--drrn_pretrained_file', type=str, default='models/models_task13-overfit/')
    parser.add_argument('--nli_model', type=str, default='roberta-large-mnli')
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()


if __name__ == '__main__':
    logger.info(f"Experiment example")
    args = parse_args()
    nli_model = init_nli_model(args)
    # loading scienceworld env
    env = ScienceWorldEnv("", "", envStepLimit=100)
    env.load(args.task, 0)

    task = "melt"
    plan_file = f"plans/plans_nl/plan_{task}_100.plan"
    drrn_model_file = "models/model_task1melt/model-steps80000-eps614.pt"

    pl = load_plan_library(plan_file)
    nli_model.reset_statistics()

    env.load(task, 26, simplificationStr="easy")
    print(env.getTaskDescription())
    agent = run_agent(plan_library=pl,
                      nli_model=nli_model,
                      env=env,
                      drrn_model_file=drrn_model_file)
    last_state = agent.belief_base.get_current_beliefs()
    bdi_score = max(last_state.score - agent.fallback_policy.score, 0)
    data = {
            'num_bdi_actions': len(agent.action_trace),
            'num_rl_actions': len(agent.fallback_policy.actions_performed),
            'plan_found': 1 if len(agent.event_trace) > 0 else 0,
            'variation': 1,
            'error': last_state.error,
            'bdi_score': bdi_score,
            'rl_score': agent.fallback_policy.score,
            'final_score': last_state.score,
            'complete': last_state.completed,
            'num_plans': len(agent.event_trace),
            'plan_library_size': len(pl.plans.keys()),
            'nli_model': args.nli_model
    }
    logger.info(json.dumps(data, indent=4))
    logger.info("Events order:")
    for trace in agent.event_trace:
        logger.info(f"=> {trace}")
    logger.info("Intentions used:")
    for trace in agent.intention_trace:
        logger.info(f"=> {trace}")

    logger.info(f"=> {data}")
