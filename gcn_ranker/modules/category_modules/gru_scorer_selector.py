from typing import Tuple, List

from torch import nn, Tensor
import torch

from gcn_ranker.modules.category_modules.scorer_selector import ScorerSelector
from gcn_ranker.modules.nlp.text2embedding import TextProcessor

try:
    from functools import cached_property
except ImportError:
    from cached_property import cached_property

class Categories:
    def __init__(self, categories: Tuple[str, ...], word_embeddings: List[Tensor]):
        self.categories = categories
        self.word_embeddings = word_embeddings
        self.category2idx = {category: idx for idx, category in enumerate(categories)}

    def __getitem__(self, category: str) -> Tensor:
        return self.word_embeddings[self.category2idx[category]]

    def idx(self, category: str) -> int:
        return self.category2idx[category]


class GruScorerSelector(nn.Module):
    def __init__(self, text_processor_config: dict, scorer_selector_config: dict, categories: Tuple[str, ...]):
        super().__init__()
        self.text_processor = TextProcessor(**text_processor_config).eval()
        self.scorer_selector = ScorerSelector(
            num_scorers=len(categories),
            **scorer_selector_config
        )
        self.gru = nn.GRU(self.text_processor.output_dim, self.scorer_selector.embed_dim)
        self.categories_str = categories

    @cached_property
    def categories(self):
        return Categories(self.categories_str, self.text_processor.process(list(self.categories_str)))


    def forward(self, category: str, features: Tensor):
        categories = torch.stack([
            self.gru(e.unsqueeze(dim=1))[1].reshape(-1) for e in self.categories.word_embeddings
        ])
        category: Tensor = categories[self.categories.idx(category)]
        scorer: Tensor = self.scorer_selector(query=category, key=categories)
        scores = torch.mm(features, scorer.reshape(-1, 1)).reshape(-1)
        return scores