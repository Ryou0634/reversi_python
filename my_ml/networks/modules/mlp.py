from typing import List, Dict

import torch

from my_ml.networks.network import Network
from my_ml.networks.modules.utils import get_mlp


@Network.register("mlp")
class MLP(Network):
    def __init__(
        self,
        encode_dims: List[int],
        activation: str = "tanh",
        dropout: float = 0.0,
        apply_activation_to_last: bool = False,
    ):
        super().__init__()

        self.mlp = get_mlp(
            encode_dims, activation=activation, dropout=dropout, apply_activation_to_last=apply_activation_to_last
        )
        self.input_size = encode_dims[0]
        self.output_size = encode_dims[-1]

        self.metrics = {}

    def get_input_size(self) -> int:
        return self.input_size

    def get_output_size(self) -> int:
        return self.output_size

    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        return self.mlp(x)
