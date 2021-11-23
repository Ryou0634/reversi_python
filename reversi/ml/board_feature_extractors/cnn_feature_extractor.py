import numpy as np

from reversi.board import ReversiBoard, Color, Position
from .board_feature_extractor import BoardFeatureExtractor


@BoardFeatureExtractor.register("cnn")
class CNNFeatureExtractor(BoardFeatureExtractor):
    def __init__(self, flatten: bool = False, add_legal_positions: bool = False):
        self.flatten = flatten
        self.add_legal_positions = add_legal_positions

    def __call__(self, board: ReversiBoard, current_color: Color) -> np.ndarray:

        board_matrix = np.full(shape=(board.size, board.size), fill_value=-1)
        for x in range(board.size):
            for y in range(board.size):
                color = board.get_color(Position(x, y))
                if color is not None:
                    board_matrix[x][y] = color.value

        player_board = board_matrix == current_color.value
        opponent_board = board_matrix == current_color.opponent.value

        features = np.stack([player_board, opponent_board]).astype(np.float32)
        if self.flatten:
            features = features.flatten()

        return features
