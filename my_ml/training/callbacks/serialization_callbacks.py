from typing import Dict
from pathlib import Path
import json
import torch
from .callbacks import EpochCallback
from my_ml.training.early_stopper import EarlyStopper
import logging

logger = logging.getLogger(__name__)


@EpochCallback.register("serialize_epoch")
class SerializeEpochMetrics(EpochCallback):
    def __init__(self, serialization_dir: str):
        self.serialization_dir = Path(serialization_dir)

    def __call__(self, metrics: Dict[str, float], epoch: int, **kwargs):
        save_path = self.serialization_dir / f"metrics_epoch_{epoch}.json"
        logger.info(f"Dump the epoch metrics to {save_path}")
        with open(save_path, "w") as f:
            json.dump(metrics, f, indent=4)


@EpochCallback.register("serialize_best_model")
class SerializeBestModel(EpochCallback):
    def __init__(self, serialization_dir: str):
        self.save_path = Path(serialization_dir) / "best.th"

    def __call__(self, model: torch.nn.Module, early_stopper: EarlyStopper, **kwargs):
        if early_stopper is not None:
            if early_stopper.current_is_best:
                logger.info(f"Best validation score. Save the model weights to {self.save_path}")
                torch.save(model.state_dict(), self.save_path)
        else:
            logger.info(f"No early stopper. Save the newest model.")
            torch.save(model.state_dict(), self.save_path)


@EpochCallback.register("serialize_every_epoch")
class SerializeEveryEpoch(EpochCallback):
    def __init__(self, serialization_dir: str, every_epoch: int = 1):
        self.serialization_dir = Path(serialization_dir)
        self.every_epoch = every_epoch

    def __call__(self, model: torch.nn.Module, epoch: int, **kwargs):
        if epoch % self.every_epoch == 0:
            save_path = self.serialization_dir / f"epoch_{epoch}.th"
            logger.info(f"Epoch {epoch}. Save the model weights to {save_path}")
            torch.save(model.state_dict(), save_path)
