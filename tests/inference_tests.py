from unittest.mock import MagicMock, Mock
import unittest
import torch

from sources.bdi_components.inference import NLIModel


class NLIModelTest(unittest.TestCase):

    def setUp(self) -> None:
        def mock_track(combinations: list[(str, str)], predicted_classes: torch.tensor):
            pass

        model = Mock()
        tokenizer = Mock()

        nli_model = NLIModel(model=model, tokenizer=tokenizer, labels2id={'entailment': 1})
        nli_model._tracking_stats = mock_track
        self.nli_model = nli_model

    def test_negative(self):
        beliefs = ['you are not seeing a ball']
        plan_contexts = ['you see a ball']
        self.nli_model._predict_nli = MagicMock(return_value=(torch.tensor([2]), torch.tensor([[0., 0., 1.]])))
        result, _ = self.nli_model.check_context_entailment(beliefs, plan_contexts)
        self.assertEqual(result, False)


if __name__ == '__main__':
    NLIModelTest.main()
