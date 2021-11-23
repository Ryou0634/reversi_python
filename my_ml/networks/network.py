from typing import List
import re
import torch
import torch.nn as nn
from registrable import Registrable, FromParams
import logging

logger = logging.getLogger(__name__)


class Network(nn.Module, Registrable):
    """
    Perform neural network computation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def get_input_size(self) -> int:
        raise NotImplementedError

    def get_output_size(self) -> int:
        raise NotImplementedError


class WeightInitializer(FromParams):
    """
    Class to initialize weights in `Network` with a serialized weight file.
    When the parameter names do not match, you can override them with regular expression.
    """

    def __init__(
        self, weight_path: str, source_module_regex: str = ".*", target_module: str = None, override_regex: str = None
    ):
        self.source_module_regex = source_module_regex
        self.weight_path = weight_path
        self.override_regex = override_regex
        self.target_module = target_module

    def initialize(self, network: nn.Module):

        if self.target_module is not None:
            network = getattr(network, self.target_module)

        weight_state_dict = torch.load(self.weight_path)

        new_state_dict = {}

        for key, weight in weight_state_dict.items():
            if not re.match(self.source_module_regex, key):
                continue

            if self.override_regex is not None:
                key = re.sub(self.source_module_regex, self.override_regex, key)
            new_state_dict[key] = weight

        logger.info(f"{network} will be initialzed with the following weights from {self.weight_path}")
        logger.info(new_state_dict.keys())

        load_result = network.load_state_dict(new_state_dict, strict=False)

        logger.info(f"Missing keys in {load_result.missing_keys}")
        logger.info(f"Unexpected keys in {load_result.unexpected_keys}")


class NetworkInitializer(FromParams):
    def __init__(self, initializers: List[WeightInitializer]):
        self.initializers = initializers

    def apply(self, network: Network):
        for initializer in self.initializers:
            initializer.initialize(network)
