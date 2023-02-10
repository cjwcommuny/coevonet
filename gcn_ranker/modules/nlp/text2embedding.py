from typing import List, Union

import torch
from torch import Tensor, nn
from transformers import AutoTokenizer, AutoModelWithLMHead


class TextProcessor(nn.Module):
    def __init__(self, model: str):
        super().__init__()
        assert model in {'bert-base-uncased', 'bert-large-uncased', 'bert-base-cased', 'bert-large-cased'}
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.word_embedding = AutoModelWithLMHead\
            .from_pretrained(model)\
            .bert.embeddings.word_embeddings

    @property
    def output_dim(self) -> int:
        return self.word_embedding.weight.shape[1]

    @torch.no_grad()
    def process(self, text: Union[str, List[str]]):
        return self(text)

    def forward(self, text: Union[str, List[str]]):
        input_ids: Union[List[int], List[List[int]]] = self.tokenizer(text)['input_ids']
        if type(input_ids[0]) == list:
            input_ids: List[Tensor] = [
                torch.tensor(ids, dtype=torch.long, device=self.word_embedding.weight.device) for ids in input_ids
            ]
            embeddings: List[Tensor] = [self.word_embedding(ids) for ids in input_ids]
            return embeddings
        else:
            input_ids: Tensor = torch.tensor(input_ids, dtype=torch.long, device=self.word_embedding.weight.device)
            embeddings: Tensor = self.word_embedding(input_ids)
            return embeddings
