from typing import Optional

import torch
from arraycontract import shape
from torch import nn, Tensor
from transformers import AutoModelWithLMHead

from gcn_ranker.modules.common import View


class TAN(nn.Module):
    def __init__(
            self,
            vocab_size: Optional[int],
            word_embed_dim: Optional[int],
            pretrained_word_embedding: Optional[str],
            video_feature_dim: int,
            hidden_dim: int,
            dropout: float
    ):
        super().__init__()
        word_embedding = nn.Embedding(vocab_size, word_embed_dim) \
            if pretrained_word_embedding is None \
            else AutoModelWithLMHead.from_pretrained(pretrained_word_embedding) \
            .bert.embeddings.word_embeddings
        #
        self.video_pipeline = nn.Sequential(
            nn.Linear(video_feature_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        self.text_pipeline = nn.Sequential(
            word_embedding,
            nn.Linear(word_embedding.weight.shape[1], hidden_dim),
            nn.Dropout(dropout),
            View(-1, 1, hidden_dim),
            nn.LSTM(hidden_dim, hidden_dim)
        )

    @shape(video=('T', 'd'), text=('N',))
    def forward(self, video: Tensor, text: Tensor):
        frame_features = self.video_pipeline(video)
        d_s = self.text_pipeline(text)[1][0].view(-1)
        raise NotImplementedError()


class MapGenerator(nn.Module):
    def __init__(self):
        super().__init__()

    @shape(X=('T', 'd'))
    def forward(self, X: Tensor):
        X_accumulated = torch.cumsum(X, dim=0).unsqueeze(0) # shape=(1,T,d)
        X = X.unsqueeze(1) # shape=(T,1,d)
        raise NotImplementedError()
