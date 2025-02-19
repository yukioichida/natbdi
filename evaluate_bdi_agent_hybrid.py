import argparse
import os.path
import random
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


def get_drrn_pretrained_info(path: str) -> pd.DataFrame:
    """
    Loads DRRN models trained using different number of episodes
    :param path: model_file path
    :return: dataframe with all trained models
    """
    model_files = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith(".pt")]
    metadata = []
    for file in model_files:
        x = re.findall("model-steps(\d*)-eps(\d*).pt", file)[0]
        steps, eps = x
        metadata.append({
            'drrn_model_file': path + file,
            'eps': eps,
            'steps': steps
        })
    models_df = pd.DataFrame(metadata).sort_values("eps")
    return models_df


def get_plan_files(task: str) -> pd.DataFrame:
    plan_files = [{"plan_file": f"plans/plans_nl/plan_{task}_100.plan", "pct_plans": 100},
                  {"plan_file": f"plans/plans_nl/plan_{task}_75.plan", "pct_plans": 75},
                  {"plan_file": f"plans/plans_nl/plan_{task}_50.plan", "pct_plans": 50},
                  {"plan_file": f"plans/plans_nl/plan_{task}_25.plan", "pct_plans": 25},
                  {"plan_file": f"plans/plans_nl/plan_{task}_0.plan", "pct_plans": 0}
                  ]
    return pd.DataFrame(plan_files).sort_values("pct_plans")


def load_experiment_info(args: argparse.Namespace) -> pd.DataFrame:
    """
    Loads all test scenarios to be executed in experiments.
    :return: Dataframe containing information of each test scenario.
    """
    plans_df = get_plan_files(args.task)
    plans_df['id'] = 0
    models_df = get_drrn_pretrained_info(args.drrn_pretrained_file)
    models_df['id'] = 0
    experiment_df = plans_df.merge(models_df, on='id', how='outer')
    if args.pct_plans:
        logger.info(f"Running with specific pct_plans: {args.pct_plans}")
        experiment_df = experiment_df[experiment_df['pct_plans'] == args.pct_plans].reset_index(drop=True)

    if args.eps:
        logger.info(f"Running with specific eps: {args.eps}")
        experiment_df = experiment_df[experiment_df['eps'] == f"{args.eps}"].reset_index(drop=True)

    return experiment_df


def random_seed(seed):
    torch.random.manual_seed(seed)
    random.seed(seed)


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
    return bdi_agent.belief_base.get_current_beliefs(), bdi_agent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='melt')
    parser.add_argument('--drrn_pretrained_file', type=str, default='models/models_task13-overfit/')
    # parser.add_argument('--nli_model', type=str, default='ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli')
    # parser.add_argument('--nli_model', type=str, default='MoritzLaurer/MiniLM-L6-mnli')
    # parser.add_argument('--nli_model', type=str, default='roberta-large-mnli')
    # parser.add_argument('--nli_model', type=str, default='gchhablani/bert-base-cased-finetuned-mnli')
    parser.add_argument('--nli_model', type=str, default='roberta-large-mnli')
    # parser.add_argument('--nli_model', type=str, default='alisawuffles/roberta-large-wanli')
    # parser.add_argument('--nli_model', type=str, default='ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli')
    parser.add_argument('--eps', type=int)
    parser.add_argument('--pct_plans', type=int)
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()


def init_nli_model(args) -> NLIModel:
    model = AutoModelForSequenceClassification \
        .from_pretrained(args.nli_model, torch_dtype=torch.float16) \
        .to(args.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.nli_model)
    nli_model = NLIModel(model=model, tokenizer=tokenizer, device=args.device)
    return nli_model


def write_results(args, results, nli_stats):
    dir = f"results/v3-{args.nli_model.replace('/', '-')}"
    if not os.path.exists(dir):
        os.makedirs(dir)
    pd.DataFrame(results).to_csv(f"{dir}/results_{args.task}.csv", index=False)
    pd.DataFrame(nli_stats).drop_duplicates().to_csv(f"{dir}/results_nli_{args.task}.csv", index=False)


if __name__ == '__main__':
    random_seed(42)
    args = parse_args()
    logger.info(args)
    # loading nli model
    nli_model = init_nli_model(args)
    # loading scienceworld env
    env = ScienceWorldEnv("", "", envStepLimit=100)
    env.load(args.task, 0)

    # Experiment parameter details
    experiment_df = load_experiment_info(args)
    results = []
    all_cases = len(experiment_df)
    nli_stats = []

    # Executing experiments
    for i, row in experiment_df.iterrows():
        logger.info(f"Experiment {i}/{all_cases} - Loading plan file: {row['plan_file']} - Results")
        pl = load_plan_library(row['plan_file'])
        nli_model.reset_statistics()
        for i, var in enumerate(env.getVariationsTest()):
            env.load(args.task, var, simplificationStr="easy")
            last_state, agent = run_agent(plan_library=pl,
                                          nli_model=nli_model,
                                          env=env,
                                          drrn_model_file=row['drrn_model_file'])

            bdi_score = max(last_state.score - agent.fallback_policy.score, 0)
            data = {
                'num_bdi_actions': len(agent.action_trace),
                'num_rl_actions': len(agent.fallback_policy.actions_performed),
                'plan_found': 1 if len(agent.event_trace) > 0 else 0,
                'variation': var,
                'error': last_state.error,
                'bdi_score': bdi_score,
                'rl_score': agent.fallback_policy.score,
                'final_score': last_state.score,
                'complete': last_state.completed,
                'num_plans': len(agent.event_trace),
                'plan_library_size': len(pl.plans.keys()),
                'plans_pct': row['pct_plans'],
                'eps': row['eps'],
                'drrn_model_file': row['drrn_model_file'],
                'nli_model': args.nli_model
            }
            results.append(data)
            logger.info(f"Results: {data}")

        nli_stats = nli_stats + nli_model.statistics

    write_results(args, results=results, nli_stats=nli_stats)
