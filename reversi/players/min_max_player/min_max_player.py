from typing import List
from reversi.board import Position, ReversiBoard, Color
from reversi.players import Player

from search_algorithm.min_max_search import MinMaxSearch
from .min_max_search import ReversiSearchNode


@Player.register("min_max")
class MinMaxPlayer(Player):
    def __init__(self, color: Color, min_max_search: MinMaxSearch):
        super().__init__(color)
        self.min_max_search = min_max_search

    def choose_position(self, current_board: ReversiBoard, legal_positions: List[Position]) -> Position:
        current_node = ReversiSearchNode(current_board, current_color=self.color, playing_color=self.color)
        return self.min_max_search.search_best_action(current_node)
