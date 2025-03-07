import torch
from torch import nn
import torch.nn.functional as F
from opacus.layers import DPLSTM  # 替换原生 LSTM

class BiLSTM(nn.Module):
    def __init__(
            self,
            input_size: int,
            rnn_size: int = 128,
            rnn_layers: int = 2,
            dropout: float = 0.3,
            is_discriminator: bool = False
    ):
        super().__init__()
        self.is_discriminator = is_discriminator

        self.lstm = DPLSTM(
            input_size=input_size,
            hidden_size=rnn_size,
            num_layers=rnn_layers,
            bidirectional=True,
            dropout=dropout if rnn_layers > 1 else 0,
            batch_first=True
        )

        if is_discriminator:
            self.classifier = nn.Sequential(
                nn.Linear(rnn_size * 4, 128),  # 混合池化后维度
                nn.LeakyReLU(0.2),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(rnn_size * 2, 256),
                nn.ReLU(),
                nn.Linear(256, input_size),
                nn.Tanh()
            )

    def forward(self, x):
        out, _ = self.lstm(x)  # [B, T, 2*rnn_size]

        if self.is_discriminator:
            avg_pool = out.mean(dim=1)
            max_pool, _ = out.max(dim=1)
            pooled = torch.cat([avg_pool, max_pool], dim=1)
            return self.classifier(pooled)
        else:
            return self.classifier(out)