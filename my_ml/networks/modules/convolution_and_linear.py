from typing import List, Dict, Tuple, Union
import math
import torch
import torch.nn as nn

from my_ml.networks.network import Network
from my_ml.networks.modules.utils import activation_dict


@Network.register("convolution_and_linear")
class ConvolutionAndLinear(Network):
    def __init__(
        self,
        input_size: Tuple[int, int, int],
        intermediate_channel_size: int,
        conv_parameters: List[Dict[str, int]],
        output_size: int,
        activation: str = "relu",
        dropout: float = 0.0,
        **kwargs
    ):
        super().__init__(**kwargs)

        conv_layers = []
        conv_output_size = input_size[1:]
        in_channels = input_size[0]
        out_channels = intermediate_channel_size
        for params in conv_parameters:
            conv_layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, **params))
            conv_layers.append(activation_dict[activation]())
            if dropout:
                conv_layers.append(nn.Dropout(p=dropout))
            conv_output_size = get_conv_output_size(conv_output_size, **params)
            in_channels = out_channels
            out_channels = out_channels * 2
        flattened_output_size = math.prod(conv_output_size) * in_channels

        self.conv_layers = nn.Sequential(*conv_layers)
        self.linear = nn.Linear(flattened_output_size, output_size)
        self.input_size = tuple(input_size)
        self.output_size = output_size

        self.metrics = {}

    def get_input_size(self) -> Tuple[int, int, int]:
        return self.input_size

    def get_output_size(self) -> int:
        return self.output_size

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        conv_output = self.conv_layers(x)
        conv_output = conv_output.flatten(start_dim=1)
        return self.linear(conv_output)


def get_conv_output_size(
    input_size: Union[int, Tuple[int, int]],
    kernel_size: Union[int, Tuple[int, int]],
    padding: Union[int, Tuple[int, int]] = 0,
    stride: Union[int, Tuple[int, int]] = 1,
) -> Tuple[int, int]:
    def _cast_to_tuple(param):
        assert isinstance(param, int) or len(param) == 2
        if isinstance(param, (tuple, list)):
            return param
        return (param, param)

    def _get_conv_output_size(input_size: int, padding: int, kernel_size: int, stride: int):
        return int(((input_size + 2 * padding - kernel_size) / stride) + 1)

    output_size = []
    for i, p, k, s in zip(
        *[_cast_to_tuple(input_size), _cast_to_tuple(padding), _cast_to_tuple(kernel_size), _cast_to_tuple(stride)]
    ):
        output_size.append(_get_conv_output_size(i, p, k, s))
    return tuple(output_size)
