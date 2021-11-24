from typing import List
from pathlib import Path
import json
import torch

from my_ml.model import Model
from reversi.board import Position, ReversiBoard, Color
from reversi.players import Player
from reversi.ml.models.move_predictor import MovePredictor
from reversi.ml.board_feature_extractors import BoardFeatureExtractor
from reversi.ml.dataset_readers.dataset_reader import index_to_position


@Player.register("ml")
class MlPlayer(Player):
    def __init__(self, serialization_dir: str, color: Color):
        super().__init__(color)

        config_file_path = Path(serialization_dir) / "config.json"
        config = json.load(open(config_file_path))

        feature_extractor_params = config["reader"]["feature_extractor"]
        self.feature_extractor = BoardFeatureExtractor.from_params(feature_extractor_params)

        model_params = config["model"]
        self.predictor = Model.from_params(model_params)
        assert isinstance(self.predictor, MovePredictor)
        weight_path = Path(serialization_dir) / "best.th"
        self.predictor.load_state_dict(torch.load(weight_path, map_location="cpu"))

    def choose_position(self, current_board: ReversiBoard, legal_positions: List[Position]) -> Position:
        feature = self.feature_extractor(current_color=self.color, board=current_board)

        output_dict = self.predictor.forward(board_feature=torch.from_numpy(feature).unsqueeze(0))
        predicted_logits = output_dict["logits"].squeeze(0)

        sorted_indices = torch.argsort(predicted_logits, dim=-1, descending=True).tolist()
        position = None
        for predicted_index in sorted_indices:
            position = index_to_position(predicted_index, size=current_board.size)
            if position in legal_positions:
                break
        assert position is not None
        return position
