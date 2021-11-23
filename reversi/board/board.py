from typing import List, Optional
from abc import ABCMeta, abstractmethod

from .color import Color
from .position import Position
from .exceptions import PositionOutOfBoundsError, InvalidPositionError

from registrable import Registrable


class ReversiBoard(Registrable, metaclass=ABCMeta):
    def __init__(self, size: int):
        self.size = size

    def _is_out_of_bounds(self, position: Position) -> bool:
        return not (0 <= position.x < self.size and 0 <= position.y < self.size)

    @abstractmethod
    def get_color(self, position: Position) -> Optional[Color]:
        raise NotImplementedError

    @abstractmethod
    def get_legal_positions(self, color: Color) -> List[Position]:
        raise NotImplementedError

    def place(self, position: Position, color: Color):
        if self._is_out_of_bounds(position):
            raise PositionOutOfBoundsError(position)

        if self.get_color(position) is not None:
            raise InvalidPositionError(position)

        self._place(position, color)

    @abstractmethod
    def _place(self, position: Position, color: Color):
        raise NotImplementedError

    @abstractmethod
    def get_num_disks(self, color: Color) -> int:
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError
