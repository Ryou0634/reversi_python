from typing import List, Union

import torch
import torch.nn as nn

activation_dict = {  # type: ignore
    "linear": lambda: lambda x: x,
    "relu": torch.nn.ReLU,
    "relu6": torch.nn.ReLU6,
    "elu": torch.nn.ELU,
    "prelu": torch.nn.PReLU,
    "leaky_relu": torch.nn.LeakyReLU,
    "threshold": torch.nn.Threshold,
    "hardtanh": torch.nn.Hardtanh,
    "sigmoid": torch.nn.Sigmoid,
    "tanh": torch.nn.Tanh,
    "log_sigmoid": torch.nn.LogSigmoid,
    "softplus": torch.nn.Softplus,
    "softshrink": torch.nn.Softshrink,
    "softsign": torch.nn.Softsign,
    "tanhshrink": torch.nn.Tanhshrink,
}


def get_mlp(dims: List[int],
            activation: str = 'relu',
            dropout: float = 0.,
            apply_activation_to_last: bool = False) -> nn.Module:
    """
    Note that we do not apply activation and dropout in the last linear layer.

    Parameters
    ----------
    dims : List[int]
    activation
    dropout

    Returns
    -------
    nn.Sequential()
    """
    num_layers = len(dims) - 1
    assert num_layers > 0

    layers = []

    for i in range(num_layers):
        in_dim, out_dim = dims[i:i + 2]
        layers += [nn.Linear(in_dim, out_dim)]

        # if this is last layer, do not apply activation and dropout
        if i == num_layers - 1 and not apply_activation_to_last:
            break

        layers += [activation_dict[activation]()]
        if dropout:
            layers += [nn.Dropout(p=dropout)]

    return nn.Sequential(*layers)


def get_conv_layers(convs: List[Union[nn.Conv2d, nn.ConvTranspose2d]],
                    activation: str = 'relu',
                    dropout: float = 0.,
                    use_batch_norm: bool = False,
                    apply_activation_to_last: bool = False) -> nn.Module:
    num_layers = len(convs)
    assert num_layers > 0

    layers = []

    for i, conv in enumerate(convs):
        layers.append(conv)

        # if this is last layer, do not apply activation and dropout
        if i == num_layers - 1 and not apply_activation_to_last:
            break

        if use_batch_norm:
            layers.append(nn.BatchNorm2d(conv.out_channels))

        layers += [activation_dict[activation]()]

        if dropout:
            layers += [nn.Dropout(p=dropout)]

    return nn.Sequential(*layers)
