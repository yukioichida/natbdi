import re
from typing import NamedTuple, Dict, List


class Plan(NamedTuple):
    triggering_event: str
    context: list[str]
    body: list[str]
    idx: int

    def plan_header(self):
        return 'if your task is to ' + self.triggering_event + ' and ' + self.context


class PlanParser:

    def parse(self, plan_content: str, idx: int) -> Plan:
        """
        Extract plan task, plan context and plan body from plan natural language representation
        :param idx: plan identifier
        :param plan_content: natural language plan
        :return Plan parsed
        """

        # TODO: see http://textx.github.io/textX/3.1/ to create more robust DSL

        re_full = r"(?<=IF your goal is to)([\S\s]*?)(?:CONSIDERING)([\S\s]*)(?:THEN)([\S\s]*)"
        re_partial = r"(?<=IF your goal is to)([\S\s]*?)(?:THEN)([\S\s]*)"  # no context

        x = re.search(re_full, plan_content)
        if x is None:
            x = re.search(re_partial, plan_content)

        if x is not None:
            groups = x.groups()
            triggering_event = self.preprocess_text(groups[0])

            if len(groups) == 3:
                context = self.preprocess_text(groups[1]).split(" AND ")
                plan_body = groups[2]
            else:
                context = []
                plan_body = groups[1]

            plan_body = [self.preprocess_text(text) for text in plan_body.split(',')]
            return Plan(triggering_event, context, plan_body, idx)
        else:
            print(f"Parse error: Plan {plan_content}")
            return None

    @staticmethod
    def preprocess_text(txt: str):
        """
        Preprocessing text removing special tokens
        :param txt: text to be processed
        :return: preprocessed text
        """
        return txt.replace('\n', ' ').replace('\t', ' ').strip()


def load_plans_from_file(file: str):
    """
    Load plans from a .plan file
    :param file: path containing the .plan file
    :return: list of plans
    """
    parser = PlanParser()
    with open(file) as f:
        plan_str = f.read()

    # plans = plan_str.split("\n--\n")
    plans = plan_str.split("\n--\n") if len(plan_str) > 0 else []
    return [parser.parse(plan, idx) for idx, plan in enumerate(plans)]


def write_plans_to_file(plan_contents: list[str], file: str):
    separator = '\n--\n'
    with open(file, 'a') as f:
        for plan in plan_contents:
            f.write(plan)
            f.write(separator)


class PlanLibrary:

    def __init__(self):
        self.plans: Dict[str, List[Plan]] = {}

    def load_plans_from_file(self, plans_file: str):
        plans_from_file = load_plans_from_file(plans_file)
        for plan in plans_from_file:
            self._add_plan(plan)

    def load_plans_from_strings(self, plans_str_list: list[str]):
        parser = PlanParser()
        for i, plan_str in enumerate(plans_str_list):
            plan = parser.parse(plan_str, i)
            self._add_plan(plan)

    def _add_plan(self, plan: Plan):
        if plan.triggering_event not in self.plans:
            self.plans[plan.triggering_event] = []
        self.plans[plan.triggering_event].append(plan)

    def get_plan_library_length(self):
        lengths = [len(plan) for key, plan in self.plans.items()]
        return sum(lengths)
