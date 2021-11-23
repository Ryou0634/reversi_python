from typing import Dict, List, Iterable, TypeVar, Iterator
import tqdm
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from my_ml.dataset_reader import DatasetReader
import itertools
from contextlib import nullcontext


import logging


from registrable import Registrable
from my_ml.training.learning_rate_scheduler import LearningRateScheduler
from my_ml.training.optimizers import Optimizer
from .early_stopper import EarlyStopper
from .callbacks.callbacks import EpochCallback, BatchCallback


logger = logging.getLogger(__name__)

A = TypeVar("A")


def lazy_groups_of(iterable: Iterable[A], group_size: int) -> Iterator[List[A]]:
    """
    Takes an iterable and batches the individual instances into lists of the
    specified size. The last list may be smaller if there are instances left over.
    """
    iterator = iter(iterable)
    while True:
        s = list(itertools.islice(iterator, group_size))
        if len(s) > 0:
            yield s
        else:
            break


def get_batch_size(batch: Dict[str, torch.Tensor]) -> int:
    return list(batch.values())[0].size(0)


class TrainerBase(Registrable):
    def train(self, *args, **kwargs):
        raise NotImplementedError

    def set_optimizer_from_params(self, optimizer_config: Dict, lr_scheduler_config: Dict):
        raise NotImplementedError


@TrainerBase.register("default")
class Trainer(TrainerBase):
    """
    A normal trainer for supervised learning.
    """

    def __init__(
        self,
        model: nn.Module,
        num_max_epochs: int = 100,
        validation_metric: str = None,
        patience: int = 3,
        device: str = "cpu",
        num_gradient_accumulation_steps: int = 1,
        epoch_callbacks: List[EpochCallback] = None,
        step_callbacks: List[BatchCallback] = None,
        no_grad_in_validation: bool = True,
    ):
        self.model = model
        self.optimizer = None
        self.lr_scheduler = None
        self.num_max_epochs = num_max_epochs

        if validation_metric:
            if validation_metric[0] == "+":
                reverse = False
            elif validation_metric[0] == "-":
                reverse = True
            else:
                raise Exception()
            self.validation_metric = validation_metric[1:]
            self.early_stopper = EarlyStopper(patience=patience, reverse=reverse)
        else:
            self.validation_metric = None
            self.early_stopper = None

        self.device = torch.device(device)
        self.model.to(self.device)

        self.num_gradient_accumulation_steps = num_gradient_accumulation_steps

        self.epoch_callbacks = epoch_callbacks or []
        self.step_callbacks = step_callbacks or []

        self.current_training_steps = 0
        self.best_state_dict = None

        self.no_grad_in_validation = no_grad_in_validation

    def set_optimizer_from_params(self, optimizer_config: Dict, lr_scheduler_config: Dict):
        self.optimizer = Optimizer.from_params(optimizer_config, model_parameters=self.model.named_parameters())
        if lr_scheduler_config is not None:
            self.lr_scheduler = LearningRateScheduler.from_params(lr_scheduler_config, optimizer=self.optimizer)
        else:
            self.lr_scheduler = None

    def train(
        self,
        train_dataset_reader: DatasetReader,
        train_data_path: str,
        validation_data_path: str,
        batch_size: int,
        validation_dataset_reader: DatasetReader = None,
    ) -> Dict[str, float]:

        result_dict = {}

        validation_dataset_reader = validation_dataset_reader or train_dataset_reader

        try:
            for epoch in range(self.num_max_epochs):

                logger.info(f"Training Epoch {epoch}")

                train_batch_generator = train_dataset_reader.generate_batches(
                    file_path=train_data_path, batch_size=batch_size, shuffle=True
                )
                training_metrics = self.run_epoch(train_batch_generator, self.optimizer, self.lr_scheduler)
                logger.info(training_metrics)
                metrics = {f"training_{k}": v for k, v in training_metrics.items()}

                if validation_dataset_reader is not None:
                    logger.info(f"Validation Epoch {epoch}")
                    validation_batch_generator = validation_dataset_reader.generate_batches(
                        file_path=validation_data_path, batch_size=batch_size, shuffle=True
                    )
                    with torch.no_grad() if self.no_grad_in_validation else nullcontext():
                        validation_metrics = self.run_epoch(validation_batch_generator)
                    logger.info(validation_metrics)
                    metrics.update({f"validation_{k}": v for k, v in validation_metrics.items()})

                    validation_score = None
                    if self.validation_metric:
                        validation_score = validation_metrics[self.validation_metric]
                        self.early_stopper.update(validation_score)

                        if self.early_stopper.current_is_best:
                            result_dict[f"best_validation_{self.validation_metric}"] = validation_score
                            result_dict[f"best_validation_epoch"] = epoch
                            self.best_state_dict = OrderedDict(
                                {key: tensor.clone().to("cpu") for key, tensor in self.model.state_dict().items()}
                            )

                        if self.early_stopper.to_stop():
                            break

                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step(validation_score)

                for callback in self.epoch_callbacks:
                    callback(epoch=epoch, metrics=metrics, model=self.model, early_stopper=self.early_stopper)
        except KeyboardInterrupt:
            logger.info("Training interrupted.")

        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)

        return result_dict

    def run_epoch(
        self,
        data_loader: DataLoader,
        optimizer: torch.optim.Optimizer = None,
        lr_scheduler: LearningRateScheduler = None,
    ):

        if optimizer is not None:
            self.model.train()
        else:
            self.model.eval()

        batch_group_generator = lazy_groups_of(data_loader, self.num_gradient_accumulation_steps)

        loss_sum_epoch = 0
        num_updates = 0
        with tqdm.tqdm(batch_group_generator) as pbar:

            for batch_group in pbar:
                if optimizer is not None:
                    optimizer.zero_grad()

                group_batch_size = sum(get_batch_size(b) for b in batch_group)
                loss_group_batch = self._forward_backward_batch_group(batch_group, is_training=optimizer is not None)
                loss_sum_epoch += loss_group_batch

                batch_metrics = {}

                if optimizer is not None:
                    optimizer.step()
                    self.current_training_steps += 1
                    if lr_scheduler is not None:
                        lr_scheduler.step_batch()

                    for i, param_group in enumerate(optimizer.state_dict()["param_groups"]):
                        batch_metrics[f"lr_param_group{i}"] = param_group["lr"]

                batch_metrics["loss"] = loss_group_batch / group_batch_size
                batch_metrics.update(self.model.get_metrics(reset=False))

                for callback in self.step_callbacks:
                    callback(step=self.current_training_steps, metrics=batch_metrics)
                pbar.set_postfix(**batch_metrics)
                num_updates += 1

        metrics_dict = {"loss": loss_sum_epoch / num_updates}

        metrics_dict.update(self.model.get_metrics(reset=True))
        metrics_dict.update(self.get_metrics(reset=True))
        return metrics_dict

    def _forward_backward_batch_group(self, batch_group: List, is_training: bool) -> float:
        loss_sum_batch_group = 0
        for batch in batch_group:

            batch = {k: tensor.to(self.device) for k, tensor in batch.items()}

            output_dict = self._forward_batch(batch)

            loss_sum_batch_group += output_dict["loss"].item() * get_batch_size(batch)

            loss = output_dict["loss"] / len(batch_group)
            if is_training:
                loss.backward()
        return loss_sum_batch_group

    def _forward_batch(self, batch: Dict[str, torch.Tensor]):
        output_dict = self.model.forward(**batch)
        return output_dict

    def get_metrics(self, reset: bool):
        return {}
