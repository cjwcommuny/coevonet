from collections import OrderedDict
from typing import List, Optional

import torch
from arraycontract import shape
from torch import nn, Tensor
from transformers import AutoModelWithLMHead

from gcn_ranker.modules.common import View, Empty
from gcn_ranker.modules.jre.jre import Jre


def max_pooling(x: Tensor, dim: int) -> Tensor:
    return torch.max(x, dim)[0]

poolings = {
    'max': max_pooling,
    'mean': torch.mean
}

def generate_sliding_windows(N: int, size: int, step: int=1) -> Tensor:
    window_starts = range(start=0, stop=N - size + 1, step=step)
    window_indexes = torch.stack([torch.arange(start=start, end=start + size) for start in window_starts])
    return window_indexes

class TextEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(input_size=in_dim, out_dim=out_dim, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_dim * 2, out_dim)

    @shape(text=('N', 'd'))
    def forward(self, text: Tensor):
        text = text.unsqueeze(1)
        output, _ = self.gru(text)
        output = output.squeeze(1)
        output = self.fc(self.dropout(output))
        return output


class MainJre(nn.Module):
    def __init__(
            self,
            vocab_size: Optional[int],
            word_embed_dim: Optional[int],
            video_feature_dim: int,
            gcn_dims: List[int],
            jre_pooling: str,
            loop_n_times: int,
            score_pooling: str,
            dropout: float,
            window_size: int,
            share: bool,
            padding_mode: str,
            pretrained_word_embedding: Optional[str],
            update_rate: float,
            text_encoder_type: str,
            use_interaction_module: bool,
            fix_language_input: bool,
            max_len: int=200
    ):
        super().__init__()
        hidden_dim = gcn_dims[0]
        self.window_size = window_size
        #
        word_embedding = nn.Embedding(vocab_size, word_embed_dim) \
            if pretrained_word_embedding is None \
            else AutoModelWithLMHead.from_pretrained(pretrained_word_embedding) \
            .bert.embeddings.word_embeddings
        #
        self.text_pipeline = nn.Sequential(OrderedDict([
            ('word_embedding', word_embedding),
            ('word_projector', nn.Linear(word_embedding.weight.shape[1], hidden_dim)),
            ('dropout', nn.Dropout(dropout)),
            ('view', View(-1, 1, hidden_dim)),
            ('gru', nn.GRU(hidden_dim, hidden_dim // 2, bidirectional=True))
        ]))
        self.video_pipeline = nn.Sequential(OrderedDict([
            ('video_projector', nn.Linear(video_feature_dim, hidden_dim)),
            ('dropout', nn.Dropout(dropout)),
            ('view', View(-1, 1, hidden_dim)),
            ('gru', nn.GRU(hidden_dim, hidden_dim // 2, bidirectional=True))
        ]))
        self.jre_loop = Jre(
            gcn_dims,
            poolings[jre_pooling],
            dropout,
            loop_n_times,
            padding_mode,
            share,
            update_rate,
            text_encoder_type,
            use_interaction_module,
            fix_language_input
        ) if loop_n_times != 0 else Empty()
        self.score_projector = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(2 * hidden_dim, 1)
        )
        self.score_pooling = poolings[score_pooling]
        self.video_position_embedding = nn.Embedding(max_len, hidden_dim)
        self.sentence_position_embedding = nn.Embedding(max_len, hidden_dim)

    @shape(video=('T', 'd'), text=('N',))
    def forward(self, video: Tensor, text: Tensor):
        """
        :return
            - window_scores: shape=(num_window,)
            - window_starts: shape=(num_window,)
        """
        T, N = video.shape[0], text.shape[0]
        video = self.video_pipeline(video)[0].view(T, -1)
        text = self.text_pipeline(text)[0].view(N, -1)
        video_position_embedding = self.video_position_embedding(torch.arange(video.shape[0], device=video.device))
        text_position_embedding = self.sentence_position_embedding(torch.arange(text.shape[0], device=text.device))
        #
        video, text = self.jre_loop(video, text, video_position_embedding, text_position_embedding)
        #
        window_starts = torch.arange(start=0, end=T - self.window_size + 1, step=1).to(video.device)
        window_features = torch.stack([self.score_pooling(video[start:start + self.window_size], dim=0) for start in window_starts])
        window_scores = self.score_projector(window_features).view(-1)
        return window_scores, window_starts
