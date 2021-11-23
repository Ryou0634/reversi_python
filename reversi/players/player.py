from typing import List
from registrable import Registrable
from reversi.board import Position, ReversiBoard, Color


class Player(Registrable):
    def __init__(self, color: Color):
        self.color = color

    def choose_position(self, current_board: ReversiBoard, legal_positions: List[Position]) -> Position:
        raise NotImplementedError
