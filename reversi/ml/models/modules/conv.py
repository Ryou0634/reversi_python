from typing import Dict, List

import torch
import torch.nn as nn

from my_ml.networks.modules.utils import get_conv_layers
from .board_encoder import BoardEncoder


@BoardEncoder.register("reversi_conv")
class ReversiConv(BoardEncoder):
    def __init__(
        self,
        num_channels: List[int],
        board_size: int = 8,
        activation: str = "relu",
        use_batch_norm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.board_size = board_size

        self.layers = get_conv_layers(
            convs=[
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1)
                for in_c, out_c in zip(num_channels[:-1], num_channels[1:])
            ],
            activation=activation,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
            apply_activation_to_last=True,
        )

        self.metrics = {}

    def get_input_size(self):
        return self.num_channels[0], self.board_size, self.board_size

    def get_output_size(self) -> int:
        return self.num_channels[-1] * self.board_size * self.board_size

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.layers(features)
        return x.flatten(start_dim=1)
