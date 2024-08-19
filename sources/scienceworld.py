import re
from typing import Callable

from scienceworld import ScienceWorldEnv

from sources.bdi_components.belief import State

"""
Scienceworld environment functions 
"""

error_messages = ["No known action matches that input.",
                  "The door is not open.",
                  "The stove appears broken, and can't be activated or deactivated."]


def load_step_function(env: ScienceWorldEnv, goal: str) -> Callable[[str], State]:
    def step_function(action: str) -> State:
        """
        Wrapper function that executes a natural language action in ScienceWorld environment
        :param action: action to be performed by the agent in the environment
        :return: state updated given the action performed
        """
        observation, reward, completed, info = env.step(action)
        error = True if observation in error_messages else False
        updated_state = parse_state(observation=observation,
                                    task=goal,
                                    info=info,
                                    completed=completed,
                                    error=error)
        return updated_state

    return step_function


def parse_goal(task_description: str) -> str:
    main_goal = task_description \
        .replace(". First, focus on the thing. Then,", "") \
        .replace(". First, focus on the substance. Then, take actions that will cause it to change its state of matter",
                 "") \
        .replace("move", "by moving") \
        .replace("Your task is to", "") \
        .replace("For compounds without a boiling point, combusting the substance is also acceptable", "") \
        .replace(".", "").strip()

    return main_goal


def parse_beliefs(observation: str,
                  look: str,
                  inventory: str) -> list[str]:
    """
    Get environment state information and generates a list of belief sentences
    :param observation: observation perceived given the last performed action
    :param look: information about objects seen
    :param inventory: inventory information
    :return: list of belief sentences
    """
    use_observation = True
    if look.split("\n")[:3] == observation.split("\n")[:3]:
        observation = ""
        use_observation = False

    x = re.search(r"([\S\s]*?)(?:In it, you see:)([\S\s]*?)(?:You also see:)([\S\s]*)", look)
    if x is None:
        x = re.search(r"([\S\s]*?)(?:Here you see:)([\S\s]*?)(?:You also see:)([\S\s]*)", look)
    groups = x.groups()

    location = groups[0]
    objects = groups[1]
    doors = groups[2]

    loc_split = [location.strip()]
    obs_split = [obs.strip() for obs in objects.split('\n') if len(obs.strip()) > 0]
    obs_split = [f"You see {obs}" for obs in obs_split]
    doors_split = [door.strip() for door in doors.split('\n') if len(door.strip()) > 0]
    inventory = inventory.replace('\n', ' ').replace('\t', '')
    if use_observation:
        return loc_split + obs_split + doors_split + [inventory, observation]
    else:
        return loc_split + obs_split + doors_split + [inventory]


def parse_state(observation: str,
                task: str,
                info: dict,
                completed: bool = False,
                error: bool = False) -> State:
    """
    ScienceWorld environment specific function to convert environment information into a State object.
    :param observation: observation received by the environment given the last performed action
    :param info: dict containing scienceworld metadata resulted from performed action
    :param task: agent's goal
    :param completed: informs whether the agent could finish the main goal
    :param error: indicates if the agent executes an action that resulted in an error
    :return:
    """
    env_state_sentences = parse_beliefs(observation=observation, look=info['look'], inventory=info['inv'])
    info['observation'] = observation
    return State(goal=task,
                 beliefs=env_state_sentences,
                 score=info['score'],
                 completed=completed,
                 error=error,
                 metadata=info)
