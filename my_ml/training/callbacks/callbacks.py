from typing import Dict
from registrable import Registrable


class BatchCallback(Registrable):
    def __call__(self, **kwargs):
        raise NotImplementedError


class EpochCallback(Registrable):

    def __call__(self, epoch: int, metrics: Dict[str, float]):
        raise NotImplementedError
