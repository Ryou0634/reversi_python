from typing import Any, Dict, List, Union

import torch

from registrable import Registrable
from .scheduler import Scheduler
from .optimizers import Optimizer


class LearningRateScheduler(Scheduler, Registrable):
    def __init__(self, optimizer: torch.optim.Optimizer, last_epoch: int = -1) -> None:
        super().__init__(optimizer, "lr", last_epoch)

    def get_values(self):
        raise NotImplementedError


class _PyTorchLearningRateSchedulerWrapper(LearningRateScheduler):
    def __init__(self, lr_scheduler: torch.optim.lr_scheduler._LRScheduler) -> None:
        self.lr_scheduler = lr_scheduler

    def get_values(self):
        return self.lr_scheduler.get_last_lr()

    def step(self, metric: float = None) -> None:
        self.lr_scheduler.step()

    def state_dict(self) -> Dict[str, Any]:
        return self.lr_scheduler.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.lr_scheduler.load_state_dict(state_dict)


class _PyTorchLearningRateSchedulerWithMetricsWrapper(_PyTorchLearningRateSchedulerWrapper):
    def step(self, metric: float = None) -> None:
        if metric is None:
            raise RuntimeError(
                "This learning rate scheduler requires "
                "a validation metric to compute the schedule and therefore "
                "must be used with a validation dataset."
            )
        self.lr_scheduler.step(metric)


@LearningRateScheduler.register("step")
class StepLearningRateScheduler(_PyTorchLearningRateSchedulerWrapper):
    """
    Registered as a `LearningRateScheduler` with name "step".  The "optimizer" argument does not get
    an entry in a configuration file for the object.
    """

    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.1, last_epoch: int = -1) -> None:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer, step_size=step_size, gamma=gamma, last_epoch=last_epoch
        )
        super().__init__(lr_scheduler)


@LearningRateScheduler.register("multi_step")
class MultiStepLearningRateScheduler(_PyTorchLearningRateSchedulerWrapper):
    """
    Registered as a `LearningRateScheduler` with name "multi_step".  The "optimizer" argument does
    not get an entry in a configuration file for the object.
    """

    def __init__(self, optimizer: Optimizer, milestones: List[int], gamma: float = 0.1, last_epoch: int = -1) -> None:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer, milestones=milestones, gamma=gamma, last_epoch=last_epoch
        )
        super().__init__(lr_scheduler)


@LearningRateScheduler.register("exponential")
class ExponentialLearningRateScheduler(_PyTorchLearningRateSchedulerWrapper):
    """
    Registered as a `LearningRateScheduler` with name "exponential".  The "optimizer" argument does
    not get an entry in a configuration file for the object.
    """

    def __init__(self, optimizer: Optimizer, gamma: float = 0.1, last_epoch: int = -1) -> None:
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=gamma, last_epoch=last_epoch)
        super().__init__(lr_scheduler)


@LearningRateScheduler.register("reduce_on_plateau")
class ReduceOnPlateauLearningRateScheduler(_PyTorchLearningRateSchedulerWithMetricsWrapper):
    """
    Registered as a `LearningRateScheduler` with name "reduce_on_plateau".  The "optimizer" argument
    does not get an entry in a configuration file for the object.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        mode: str = "min",
        factor: float = 0.1,
        patience: int = 10,
        verbose: bool = False,
        threshold_mode: str = "rel",
        threshold: float = 1e-4,
        cooldown: int = 0,
        min_lr: Union[float, List[float]] = 0,
        eps: float = 1e-8,
    ) -> None:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            verbose=verbose,
            threshold_mode=threshold_mode,
            threshold=threshold,
            cooldown=cooldown,
            min_lr=min_lr,
            eps=eps,
        )
        super().__init__(lr_scheduler)


@LearningRateScheduler.register("slanted_triangular")
class SlantedTriangularLearningRateScheduler(LearningRateScheduler):
    """
    https://arxiv.org/pdf/1801.06146.pdf
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        cut_frac: float = 0.1,
        ratio: int = 32,
        last_epoch: int = -1,
    ):
        self.total_steps = total_steps
        self.cut_frac = cut_frac
        self.cut = int(total_steps * cut_frac)
        self.ratio = ratio
        self.t = 0
        self.base_lrs = list(map(lambda group: group["lr"], optimizer.param_groups))
        super().__init__(optimizer, last_epoch)

    def get_values(self):
        t = min(self.total_steps, self.t)
        if t < self.cut:
            p = t / self.cut
        else:
            p = 1 - (t - self.cut) / (self.cut * (1 / self.cut_frac - 1))

        lrs = [base_lr * (1 + (p * (self.ratio - 1))) / self.ratio for base_lr in self.base_lrs]
        return lrs

    def step(self, metric: float = None) -> None:
        pass

    def step_batch(self, batch_num_total: int = None) -> None:
        if batch_num_total is not None:
            self.t = batch_num_total
        else:
            self.t += 1

        for param_group, lr in zip(self.optimizer.param_groups, self.get_values()):
            param_group["lr"] = lr
