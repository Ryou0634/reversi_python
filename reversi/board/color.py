from enum import Enum, auto


class Color(Enum):
    WHITE = auto()
    BLACK = auto()

    @property
    def opponent(self):
        if self == self.WHITE:
            return self.BLACK
        else:
            return self.WHITE

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name
