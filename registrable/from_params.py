import inspect
import typing
from typing import TypeVar, Type, Dict


T = TypeVar("T", bound="FromParams")


def instantiate_class_from_params(cls: Type[T], params: Dict, **kwargs) -> T:
    for name, param in inspect.signature(cls).parameters.items():
        child_class = param.annotation
        if name not in params:
            continue
        if isinstance(child_class, typing._GenericAlias):
            # special case for List[Registrable]
            contained_class = typing.get_args(child_class)[0]
            if issubclass(contained_class, FromParams):
                child_instance = [contained_class.from_params(c) for c in params[name]]
                params[name] = child_instance
        # Recursively call the submodule's from_params
        elif issubclass(child_class, FromParams):
            child_instance = child_class.from_params(params[name])
            params[name] = child_instance
    return cls(**params, **kwargs)


class FromParams:
    """
    Mixin to give a from_params method to classes.
    """

    @classmethod
    def from_params(cls: Type[T], params: Dict, **kwargs) -> T:
        return instantiate_class_from_params(cls, params, **kwargs)
