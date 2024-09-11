import abc

import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer, AutoConfig

from sources.cl_nli.model import SimCSE


class EncoderModel:

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    @abc.abstractmethod
    def encode(self,
               text: list[str]) -> torch.Tensor:
        raise NotImplementedError()

    def encode_batch(self,
                     texts: list[str],
                     max_size: int = 25,
                     include_cls: bool = True) -> torch.Tensor:
        pad_token = self.tokenizer.pad_token
        if include_cls:
            cls_token = self.tokenizer.cls_token
            texts = [cls_token] + texts
        pad_size = max_size - len(texts)
        padding = [pad_token for _ in range(pad_size)]
        texts = texts + padding
        batch_embeddings = self.encode(texts)
        return batch_embeddings


class HFEncoderModel(EncoderModel):

    def __init__(self, hf_model_name: str, device: str):
        super().__init__(tokenizer=AutoTokenizer.from_pretrained(hf_model_name))
        self.encoder_model = AutoModel.from_pretrained(hf_model_name).eval().to(device)
        encoder_config = AutoConfig.from_pretrained(hf_model_name)
        self.embedding_dim = encoder_config.hidden_size

    def encode(self, texts: list[str]) -> torch.Tensor:
        with torch.no_grad():
            device = self.encoder_model.device
            tokenized_text = self.tokenizer(texts,
                                            padding='longest',
                                            truncation=True,
                                            return_tensors='pt').to(device)
            embeddings = self.encoder_model(**tokenized_text).pooler_output.unsqueeze(0).detach()
            return embeddings


class CustomSimCSEModel(EncoderModel):

    def __init__(self, encoder_model: SimCSE):
        hf_model_name = encoder_model.hparams['hf_model_name']
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        super().__init__(tokenizer)
        self.encoder_model = encoder_model

    def encode(self, texts: list[str]) -> torch.Tensor:
        device = self.encoder_model.device
        tokenized_text = self.tokenizer(texts,
                                        padding='longest',
                                        truncation=True,
                                        return_tensors='pt').to(device)
        embeddings = self.encoder_model.encode(tokenized_text).detach()  # batch axis
        return embeddings
