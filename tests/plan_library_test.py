from unittest.mock import MagicMock, Mock
import unittest
import torch

from sources.bdi_components.plans import Plan, PlanLibrary, PlanParser


class PlanParserTest(unittest.TestCase):

    def test_plan_with_no_context(self):
        goal = "boil water"
        plan_body = ["do something"]
        plan_test = f"""
            IF your goal is to {goal}
            THEN
            {' '.join(plan_body)}
        """

        plan_parser = PlanParser()
        plan = plan_parser.parse(plan_test, 0)
        # goal addition
        self.assertEqual(plan.triggering_event, goal)
        self.assertEqual(plan.context, [])
        self.assertEqual(plan.body, plan_body)


    def test_plan_with_context(self):
        goal = "take thermometer"
        plan_body = ["pick up thermometer"]
        plan_contexts = ["you are in the kitchen",
                         "you see a thermometer",
                         "you don't have a thermometer in your inventory"]
        plan_test = f"""
            IF your goal is to {goal}
            CONSIDERING {' AND '.join(plan_contexts)}
            THEN
            {' '.join(plan_body)}
        """

        plan_parser = PlanParser()
        plan = plan_parser.parse(plan_test, 0)
        # goal addition
        self.assertEqual(plan.triggering_event, goal)
        self.assertEqual(plan.context, plan_contexts)
        self.assertEqual(plan.body, plan_body)

    def test_plan_no_goal(self):
        "the plan should have at least a goal"
        plan_body = ["pick up thermometer"]
        plan_contexts = ["you are in the kitchen",
                         "you see a thermometer",
                         "you don't have a thermometer in your inventory"]
        plan_test = f"""
            CONSIDERING {' AND '.join(plan_contexts)}
            THEN
            {' '.join(plan_body)}
        """

        plan_parser = PlanParser()
        plan = plan_parser.parse(plan_test, 0)
        # goal addition
        self.assertIsNone(plan)


if __name__ == '__main__':
    PlanParserTest.main()
