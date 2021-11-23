from typing import List, Dict, Tuple, Union
import torch
import torch.nn as nn

from my_ml.networks.network import Network
from my_ml.networks.modules.utils import activation_dict


@Network.register("deconvolution")
class Deconvolution(Network):
    def __init__(
        self,
        input_size: int,
        intermediate_channel_size: int,
        out_channel_size: int,
        deconv_parameters: List[Dict[str, int]],
        activation: str = "relu",
        last_activation: str = "sigmoid",
        dropout: float = 0.0,
    ):
        super().__init__()

        conv_layers = []
        in_channels = input_size
        out_channels = intermediate_channel_size
        deconv_output_size = (1, 1)
        for i, params in enumerate(deconv_parameters):
            if i == len(deconv_parameters) - 1:
                activation_function = activation_dict[last_activation]()
                out_channels = out_channel_size
            else:
                activation_function = activation_dict[activation]()

            conv_layers.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, **params))
            conv_layers.append(activation_function)

            if dropout:
                conv_layers.append(nn.Dropout(p=dropout))
            deconv_output_size = get_deconv_output_size(deconv_output_size, **params)
            in_channels = out_channels
            out_channels = out_channels // 2
        output_size = (out_channel_size,) + deconv_output_size

        self.conv_layers = nn.Sequential(*conv_layers)
        self.input_size = input_size
        self.output_size = output_size

    def get_input_size(self) -> int:
        return self.input_size

    def get_output_size(self) -> Tuple[int, int, int]:
        return self.output_size

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = x.unsqueeze(-1).unsqueeze(-1)
        return self.conv_layers(x)


def get_deconv_output_size(
    input_size: Union[int, Tuple[int, int]],
    kernel_size: Union[int, Tuple[int, int]],
    padding: Union[int, Tuple[int, int]] = 0,
    stride: Union[int, Tuple[int, int]] = 1,
) -> Tuple[int, int]:
    def _cast_to_tuple(param):
        assert isinstance(param, int) or len(param) == 2
        if isinstance(param, tuple):
            return param
        return (param, param)

    def _get_deconv_output_size(input_size: int, padding: int, kernel_size: int, stride: int):
        # output_size = strides * (input_size-1) + kernel_size - 2*padding
        return int(stride * (input_size - 1) + kernel_size - 2 * padding)

    output_size = []
    for i, p, k, s in zip(
        *[_cast_to_tuple(input_size), _cast_to_tuple(padding), _cast_to_tuple(kernel_size), _cast_to_tuple(stride)]
    ):
        output_size.append(_get_deconv_output_size(i, p, k, s))
    return tuple(output_size)
