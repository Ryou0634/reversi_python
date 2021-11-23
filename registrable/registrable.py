import importlib
import logging
import pkgutil
import sys
from collections import defaultdict
from typing import TypeVar, Type, Callable, Dict
from abc import ABCMeta

from .from_params import FromParams, instantiate_class_from_params

T = TypeVar("T", bound="Registrable")

logger = logging.getLogger(__name__)


def import_submodules(package_name: str) -> None:
    """
    Import all submodules under the given package.
    Primarily useful for `Registrable` classes to find the classes that should be in its registry.
    """
    importlib.invalidate_caches()

    # For some reason, python doesn't always add this by default to your path, but you pretty much
    # always want it when using `--include-package`.  And if it's already there, adding it again at
    # the end won't hurt anything.
    sys.path.append("")

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


class Registrable(FromParams, metaclass=ABCMeta):
    """
    Any class that inherits from `Registrable` gains access to a named registry for its
    subclasses. To register them, just decorate them with the classmethod
    `@BaseClass.register(name)`.

    Adapted from (https://docs.allennlp.org/main/api/common/registrable/)
    """

    _registry: Dict[Type, Dict[str, Type]] = defaultdict(dict)

    @classmethod
    def register(cls: Type[T], name: str, exist_ok: bool = False):

        registry = Registrable._registry[cls]

        def add_subclass_to_registry(subclass: Type[T]):

            # Add to registry, raise an error if key has already been used.
            if name in registry:
                if exist_ok:
                    message = (
                        f"{name} has already been registered as {registry[name][0].__name__}, but "
                        f"exist_ok=True, so overwriting with {cls.__name__}"
                    )
                    logger.info(message)
                else:
                    message = (
                        f"Cannot register {name} as {cls.__name__}; "
                        f"name already in use for {registry[name][0].__name__}"
                    )
                    raise Exception(message)
            registry[name] = subclass
            return subclass

        return add_subclass_to_registry

    @classmethod
    def by_name(cls: Type[T], name: str) -> Callable[..., T]:
        """
        Returns a callable function that constructs an argument of the registered class.
        """
        logger.debug(f"instantiating registered subclass {name} of {cls}")
        if name not in Registrable._registry[cls]:
            raise Exception(f"``{name}`` not found for {cls}")
        return Registrable._registry[cls][name]

    @classmethod
    def from_params(cls: Type[T], config: Dict, **kwargs) -> T:

        if "type" in config:
            name = config.pop("type")
            logger.debug(f"instantiating registered subclass {name} of {cls} from config")
            logger.debug(config)
            this_class = Registrable._registry[cls][name]
        else:
            this_class = cls

        return instantiate_class_from_params(this_class, config, **kwargs)
