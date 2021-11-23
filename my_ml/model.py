import torch.nn as nn
from registrable import Registrable


class Model(Registrable, nn.Module):
    def get_metrics(self, reset: bool):
        return {}
