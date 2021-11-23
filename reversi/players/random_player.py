from typing import List
import random


from reversi.board import Position, ReversiBoard
from .player import Player


@Player.register("random")
class RandomPlayer(Player):
    def choose_position(self, current_board: ReversiBoard, legal_positions: List[Position]) -> Position:
        return random.choice(legal_positions)
