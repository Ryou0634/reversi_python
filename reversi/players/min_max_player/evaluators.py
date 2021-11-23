from reversi.board import Position
from search_algorithm.min_max_search import SearchNodeEvaluator

from .min_max_search import ReversiSearchNode

score_matrix = [
    [120, -20, 20, 5, 5, 20, -20, 120],
    [-20, -40, -5, -5, -5, -5, -40, -20],
    [20, -5, 15, 3, 3, 15, -5, 20],
    [5, -5, 3, 3, 3, 3, -5, 5],
    [5, -5, 3, 3, 3, 3, -5, 5],
    [20, -5, 15, 3, 3, 15, -5, 20],
    [-20, -40, -5, -5, -5, -5, -40, -20],
    [120, -20, 20, 5, 5, 20, -20, 120],
]


@SearchNodeEvaluator.register("reversi_manual")
class ReversiManualEvaluator(SearchNodeEvaluator):
    def __init__(self, yaocho: bool = False):
        self.yaocho = yaocho

    def evaluate(self, node: ReversiSearchNode) -> float:
        board = node.board
        score = 0
        for x in range(board.size):
            for y in range(board.size):
                position = Position(x, y)
                if board.get_color(position) == node.playing_color:
                    score += score_matrix[x][y]
                else:
                    score -= score_matrix[x][y]
        if self.yaocho:
            score = - score
        return score
