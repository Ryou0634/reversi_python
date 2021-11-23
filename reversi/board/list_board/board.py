from typing import Optional, List


from dataclasses import dataclass

from reversi.board.color import Color
from reversi.board.position import Position
from reversi.board.board import ReversiBoard
from reversi.board.exceptions import InvalidPositionError


@dataclass
class Direction:
    x_offset: int
    y_offset: int

    def __post_init__(self):
        assert -1 <= self.x_offset <= 1
        assert -1 <= self.y_offset <= 1
        assert not (self.x_offset == self.y_offset == 0)

    def __add__(self, other) -> Position:
        if not isinstance(other, Position):
            raise TypeError(type(other))
        return Position(other.x + self.x_offset, other.y + self.y_offset)


@ReversiBoard.register("list")
class ListBoard(ReversiBoard):
    def __init__(self, size: int = 8):
        super().__init__(size=size)
        self.cells = [[None for _ in range(self.size)] for _ in range(self.size)]

        self.reset()

    def _set_disk(self, position: Position, color: Color):
        self.cells[position.x][position.y] = color

    @staticmethod
    def _generate_directions():
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                if x == y == 0:
                    continue
                yield Direction(x, y)

    def get_color(self, position: Position) -> Optional[Color]:
        return self.cells[position.x][position.y]

    def _place(self, position: Position, color: Color):
        # flip other disks
        num_total_flipped_disk = 0
        for direction in self._generate_directions():
            num_bounded_disks = self._count_bounded_disks(position, color, direction)
            num_total_flipped_disk += num_bounded_disks
            current_position = position
            for _ in range(num_bounded_disks):
                current_position = direction + current_position
                current_color = self.get_color(current_position)
                self._set_disk(current_position, current_color.opponent)

        if num_total_flipped_disk == 0:
            raise InvalidPositionError(position)

        self._set_disk(position, color)

    def get_num_disks(self, color: Color) -> int:
        return sum(c == color for cs in self.cells for c in cs)

    def get_legal_positions(self, color: Color) -> List[Position]:
        legal_positions = []
        for x in range(self.size):
            for y in range(self.size):
                position = Position(x, y)
                if self.get_color(position) is not None:
                    continue
                if any(self._count_bounded_disks(position, color, d) for d in self._generate_directions()):
                    legal_positions.append(position)
        return legal_positions

    def _count_bounded_disks(self, position: Position, color: Color, direction: Direction) -> int:
        num_bounded_disks = 0
        current_position = position

        # search toward the direction
        while True:
            current_position = direction + current_position
            if self._is_out_of_bounds(current_position):
                return 0

            if self.get_color(current_position) is None:
                return 0
            elif self.get_color(current_position) != color:
                # if you find an opponent disk, increase num_bounded_disks.
                num_bounded_disks += 1
            elif self.get_color(current_position) == color:
                # if you find one of your disks, return num_bounded_disks.
                return num_bounded_disks

    def reset(self):
        center = self.size // 2

        # set the initial disks
        self._set_disk(Position(center - 1, center - 1), Color.WHITE)
        self._set_disk(Position(center, center), Color.WHITE)
        self._set_disk(Position(center - 1, center), Color.BLACK)
        self._set_disk(Position(center, center - 1), Color.BLACK)

    def draw_screen(self) -> str:
        return_str = "  a b c d e f g h"
        for row in range(self.size):
            return_str += "\n" + str(row + 1)
            for col in range(self.size):
                if self.cells[Color.Black][row][col]:
                    cell_state = "●"
                elif self.cells[Color.White][row][col]:
                    cell_state = "○"
                else:
                    cell_state = "-"
                return_str += " " + str(cell_state)
        return return_str
