import string
from dataclasses import dataclass
from .exceptions import InvalidPositionError

X_TO_MOVE = [str(i) for i in range(1, 10)]
Y_TO_MOVE = string.ascii_lowercase

MOVE_TO_X = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5, "7": 6, "8": 7}
MOVE_TO_Y = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7}


@dataclass
class Position:
    x: int
    y: int

    def __hash__(self):
        return hash((self.x, self.y))

    def __str__(self):
        return Y_TO_MOVE[self.y] + X_TO_MOVE[self.x]

    @classmethod
    def from_move(cls, move: str) -> "Position":
        y_move, x_move = move
        try:
            return cls(x=MOVE_TO_X[x_move], y=MOVE_TO_Y[y_move])
        except KeyError:
            raise InvalidPositionError()
