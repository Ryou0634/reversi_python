import importlib
import sys
import pkgutil

import _jsonnet
import json
import random
import numpy
import torch

from my_ml.model import Model


def read_model_from_params(config_path: str, weights_path: str = None, include_package: str = "src"):
    import_submodules(include_package)
    config = json.loads(_jsonnet.evaluate_file(config_path))

    model = Model.from_params(config["model"])

    if weights_path is not None:
        model.load_state_dict(torch.load(weights_path))
    model.eval()
    return model


def import_submodules(package_name: str) -> None:
    """
    Import all submodules under the given package.
    Primarily useful so that people using AllenNLP as a library
    can specify their own custom packages and have their custom
    classes get loaded and registered.
    """
    importlib.invalidate_caches()

    # For some reason, python doesn't always add this by default to your path, but you pretty much
    # always want it when using `--include-package`.  And if it's already there, adding it again at
    # the end won't hurt anything.
    sys.path.append(".")

    # Import at top level
    module = importlib.import_module(package_name)
    path = getattr(module, "__path__", [])
    path_string = "" if not path else path[0]

    # walk_packages only finds immediate children, so need to recurse.
    for module_finder, name, _ in pkgutil.walk_packages(path):
        # Sometimes when you import third-party libraries that are on your path,
        # `pkgutil.walk_packages` returns those too, so we need to skip them.
        if path_string and module_finder.path != path_string:
            continue
        subpackage = f"{package_name}.{name}"
        import_submodules(subpackage)


def set_random_seed(seed: int = None):

    if seed is not None:
        random.seed(seed)
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        # Seed all GPUs with the same seed if available.
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
