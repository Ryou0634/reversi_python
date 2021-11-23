from typing import Dict

from my_ml.training.metrics import CategoricalAccuracy
from my_ml.model import Model
from .modules.board_encoder import BoardEncoder

import torch
import torch.nn as nn


@Model.register("move_predictor")
class MovePredictor(Model):
    def __init__(self, board_encoder: BoardEncoder, board_size: int = 8):
        super().__init__()
        self.board_encoder = board_encoder

        self.output_projection_layer = nn.Linear(self.board_encoder.get_output_size(), board_size ** 2)

        self.board_size = board_size
        self.criterion = nn.CrossEntropyLoss()
        accuracy = CategoricalAccuracy()

        self.metrics = {"accuracy": accuracy}

    def forward(self, board_feature: torch.Tensor, move: torch.Tensor = None):

        logits = self.output_projection_layer(self.board_encoder(board_feature))
        prediction = logits.argmax(dim=1)

        output_dict = {"prediction": prediction, "logits": logits}

        if move is not None:
            self.metrics["accuracy"](logits, move)
            output_dict["loss"] = self.criterion(logits, move)

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        output_dict = {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}
        return output_dict
