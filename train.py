from typing import Dict, List, Any
import click
import itertools

from my_ml.utils import import_submodules, set_random_seed
from my_ml.model import Model
from my_ml.dataset_reader import DatasetReader
from my_ml.training.callbacks import EpochCallback, BatchCallback
from my_ml.training.callbacks.serialization_callbacks import (
    SerializeBestModel,
    SerializeEpochMetrics,
    SerializeEveryEpoch,
)
from my_ml.training import TrainerBase

import _jsonnet
import json
from pathlib import Path

import logging

logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_parameter_json(parameters_path: str):
    parameters = json.loads(_jsonnet.evaluate_file(parameters_path))
    keys = parameters.keys()
    values = [parameters[k] for k in keys]
    return [dict(zip(keys, combinations)) for combinations in itertools.product(*values)]


def override_config(config: Dict, parameters: Dict[str, Any]):
    def replace_with_keys(dictionary: Dict, keys: List, value: Any):
        k = keys.pop(0)
        if len(keys) == 0:
            dictionary[k] = value
        else:
            replace_with_keys(dictionary[k], keys, value)

    for keys, value in parameters.items():
        keys = keys.split(".")
        replace_with_keys(config, keys, value)
    return config


def run_train(
    config: Dict, device: str, epoch_callbacks: List[EpochCallback] = None, step_callbacks: List[BatchCallback] = None
) -> Dict[str, float]:
    train_reader = DatasetReader.from_params(config["reader"])
    validation_reader = train_reader
    if "validation_reader" in config:
        validation_reader = DatasetReader.from_params(config["validation_reader"])
    model = Model.from_params(config["model"])

    trainer = TrainerBase.from_params(
        config["trainer"],
        model=model,
        validation_metric=config.get("validation_metric", None),
        device=device,
        epoch_callbacks=epoch_callbacks,
        step_callbacks=step_callbacks,
    )

    trainer.set_optimizer_from_params(config["optimizer"], config.get("lr_scheduler", None))

    try:
        result_dict = trainer.train(
            train_dataset_reader=train_reader,
            train_data_path=config["train_data_path"],
            validation_data_path=config["validation_data_path"],
            validation_dataset_reader=validation_reader,
            batch_size=config["batch_size"],
        )
    except KeyboardInterrupt:
        logger.info("Training interrupted.")
        result_dict = {}

    return result_dict


@click.command()
@click.argument("config-path", type=click.Path(exists=True))
@click.argument("serialization-dir", type=click.Path(exists=False))
@click.option("--device", type=str, default="cpu")
@click.option("--seed", type=int, default=0)
@click.option("--serialize-every", type=int, default=0)
@click.option("--parameters-path", type=click.Path(exists=True))
def train(config_path: str, serialization_dir: str, device: str, seed: int, serialize_every: int, parameters_path: str):

    # Read config
    import_submodules("my_ml")
    import_submodules("reversi")

    set_random_seed(seed)

    if parameters_path:
        params = parse_parameter_json(parameters_path)
    else:
        params = [{}]

    logger.info(f"Train with {len(params)} hyper parameters.")
    for i, hyper_params in enumerate(params):

        config = json.loads(_jsonnet.evaluate_file(str(config_path)))
        if hyper_params:
            config = override_config(config, hyper_params)

        # prepare serialization_dir
        config_path = Path(config_path)
        save_path = Path(serialization_dir) / f"params{i}"
        save_path.mkdir(parents=True, exist_ok=True)
        with open(save_path / "config.json", "w") as f:
            json.dump(config, f, indent=4)

        epoch_callbacks = [SerializeEpochMetrics(save_path), SerializeBestModel(save_path)]
        if serialize_every:
            epoch_callbacks.append(SerializeEveryEpoch(save_path, serialize_every))
        if "epoch_callbacks" in config["trainer"]:
            callback_configs = config["trainer"].pop("epoch_callbacks")
            for c in callback_configs:
                epoch_callbacks.append(EpochCallback.from_params(c, serialization_dir=save_path))

        result_dict = run_train(config, device=device, epoch_callbacks=epoch_callbacks)

        logger.info(result_dict)

        with open(save_path / "metrics.json", "w") as f:
            json.dump(result_dict, f, indent=4)


def main():
    train()


if __name__ == "__main__":
    main()
