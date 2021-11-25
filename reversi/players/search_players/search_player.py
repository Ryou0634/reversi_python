from typing import List
import copy

from reversi.board import Position, ReversiBoard, Color
from reversi.players import Player

from search_algorithm import SearchAlgorithm

from search_algorithm.min_max_search import TreeNode

SKIP_ACTION = None


def board_to_state(board: ReversiBoard, current_color: Color) -> TreeNode:
    return ReversiSearchNode(board, current_color, current_color)


@TreeNode.register("reversi")
class ReversiSearchNode(TreeNode):
    def __init__(self, board: ReversiBoard, current_color: Color, playing_color: Color):
        self.board = board
        self.current_color = current_color
        self.playing_color = playing_color

    def get_valid_actions(self) -> List[Position]:
        legal_positions = self.board.get_legal_positions(self.current_color)
        return legal_positions

    def get_next_node(self, position: Position) -> TreeNode:
        new_board = copy.deepcopy(self.board)
        new_board.place(position, self.current_color)
        return ReversiSearchNode(new_board, self.current_color.opponent, self.playing_color)

    @property
    def is_opponent_turn(self) -> bool:
        return self.current_color == self.playing_color.opponent

    @property
    def is_terminal(self) -> bool:
        legal_positions = self.board.get_legal_positions(self.current_color) + self.board.get_legal_positions(
            self.current_color
        )
        return len(legal_positions) == 0


@Player.register("search")
class MinMaxPlayer(Player):
    def __init__(self, color: Color, search_algorithm: SearchAlgorithm):
        super().__init__(color)
        self.search_algorithm = search_algorithm

    def choose_position(self, current_board: ReversiBoard, legal_positions: List[Position]) -> Position:
        current_node = ReversiSearchNode(current_board, current_color=self.color, playing_color=self.color)
        return self.search_algorithm.search_best_action(current_node)
