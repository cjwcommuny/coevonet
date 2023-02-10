from typing import Optional

import torch
from arraycontract import shape
from torch import nn, Tensor
from torchtools.modules import AdditiveAttention
from transformers import AutoModelWithLMHead

from gcn_ranker.metrics.smooth_l1 import smooth_l1_norm
from gcn_ranker.modules.common import View


class ABLR(nn.Module):
    """
    Yuan, Yitian, Tao Mei, and Wenwu Zhu. “To find where you talk: Temporal sentence localization in video with attention based location regression.” Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 33. 2019.
    """
    def __init__(
            self,
            window_size: int,
            vocab_size: Optional[int],
            word_embed_dim: Optional[int],
            pretrained_word_embedding: Optional[str],
            video_feature_dim: int,
            hidden_dim: int,
            dropout: float
    ):
        super().__init__()
        self.window_size = window_size
        assert hidden_dim % 2 == 0
        word_embedding = nn.Embedding(vocab_size, word_embed_dim) \
            if pretrained_word_embedding is None \
            else AutoModelWithLMHead.from_pretrained(pretrained_word_embedding) \
            .bert.embeddings.word_embeddings
        #
        self.video_pipeline = nn.Sequential(
            nn.Linear(video_feature_dim, hidden_dim),
            nn.Dropout(dropout),
            View(-1, 1, hidden_dim),
            nn.LSTM(hidden_dim, hidden_dim // 2, bidirectional=True)
        )
        self.text_pipeline = nn.Sequential(
            word_embedding,
            nn.Linear(word_embedding.weight.shape[1], hidden_dim),
            nn.Dropout(dropout),
            View(-1, 1, hidden_dim),
            nn.LSTM(hidden_dim, hidden_dim // 2, bidirectional=True)
        )
        #
        self.attention1 = AdditiveAttention(hidden_dim, hidden_dim, hidden_dim)
        self.attention2 = AdditiveAttention(hidden_dim, hidden_dim, hidden_dim)
        self.attention3 = AdditiveAttention(hidden_dim, hidden_dim, hidden_dim)
        #
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
            nn.Sigmoid()
        )

    @shape(video=('T', 'd'), text=('N',))
    def forward(self, video: Tensor, text: Tensor):
        """
        :param video:
        :param text:
        :return: shape=(2,), (start, end)
        """
        T, N = video.shape[0], text.shape[0]
        frame_features = self.video_pipeline(video)[0].view(T, -1)
        sentence_features = self.text_pipeline(text)[0].view(N, -1)
        sentence_mean_features = torch.mean(sentence_features, dim=0)
        attr1 = self.attention1(sentence_mean_features.unsqueeze(0), frame_features.unsqueeze(0)) # shape=(1, hidden_d)
        attr2 = self.attention2(attr1, sentence_features.unsqueeze(0)) # shape=(1, hidden_d)
        attr3 = self.attention3(attr2, frame_features.unsqueeze(0)) # shape=(1, hidden_d)
        feature_fused = torch.cat((attr2, attr3), dim=1)
        position = self.fc(feature_fused).view(2)
        return position
