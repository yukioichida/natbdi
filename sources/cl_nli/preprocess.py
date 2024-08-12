import abc

from datasets import Dataset
from transformers import PreTrainedTokenizer
import random

nli_objective = {
        'entailment': 1.,
        'neutral': 0.,
        #'contradiction': 0
        'contradiction': -1.,
}


class DatasetPreprocessor:
    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 sentence_cols: list[str],
                 cols_to_ignore: list[str],
                 label_col: str = 'gold_label'):
        self.tokenizer = tokenizer
        self.cols_to_ignore = cols_to_ignore
        self.sentence_cols = sentence_cols
        self.norm_sentence_cols = [f"s{i + 1}" for i in range(len(sentence_cols))]
        self.label_col = label_col

    def preprocess(self, dataset: Dataset) -> Dataset:
        def tokenize_fragments(example):
            example = self.init_features(example)
            example = self.normalize_cols(example)
            return example

        dataset = dataset.map(tokenize_fragments,
                              batched=True,
                              load_from_cache_file=False,
                              remove_columns=self.cols_to_ignore)
        dataset.set_format('torch')
        dataset = dataset.remove_columns(self.sentence_cols)
        return dataset

    def normalize_cols(self, example):
        """
        Standardizes column name for all dataset (sentence1, sentence2, sentence3)
        :param example:
        :return: example with normalized columns
        """
        norm_cols = []
        for col, norm_col in zip(self.sentence_cols, self.norm_sentence_cols):
            example[norm_col] = example[col]
        return example

    @abc.abstractmethod
    def init_features(self, example):
        pass


class PairwiseDatasetReader(DatasetPreprocessor):

    def init_features(self, example):
        example['nli_objective'] = [nli_objective[s] for s in example[self.label_col]]
        return example


class SemanticFragmentsDatasetReader(DatasetPreprocessor):

    def init_features(self, example):
        # sentence to be approximated
        entailment_sentences = [example['sentence2'][i]
                                if example['gold_label'][i] == 'entailment'
                                else example['sentence1'][i]
                                for i in range(len(example['sentence1']))]
        example['entailment'] = entailment_sentences
        # sentence to be separated
        example['contradiction'] = [example['sentence2'][i]
                                    if example['gold_label'][i] != 'entailment'
                                    else ' '
                                    for i in range(len(example['sentence1']))]
        example['nli_weight'] = [1. if example['gold_label'] == 'contradiction' else 0 for i in
                                 range(len(example['sentence1']))]
        return example


class SimCSEDatasetReader(DatasetPreprocessor):
    def init_features(self, example):
        # every negative sample are actual contradiction pair
        example['nli_weight'] = [1.] * len(example['sent0'])
        return example
