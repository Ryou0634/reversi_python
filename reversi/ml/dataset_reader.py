import glob
import tqdm
from torch.utils.data import DataLoader

from my_ml.dataset_reader import DatasetReader
from reversi.game_engine import ReversiGameEngine
from reversi.board import Position, InvalidPositionError
from .wtb_file_parser import parse_wtb_file
from .board_feature_extractors import BoardFeatureExtractor

import logging

logger = logging.getLogger(__name__)


def position_to_index(position: Position, size: int):
    return position.x * size + position.y


@DatasetReader.register("reversi_move_prediction")
class ReversiDatasetReader(DatasetReader):
    def __init__(self, feature_extractor: BoardFeatureExtractor):
        self.feature_extractor = feature_extractor

        self._game_engine = ReversiGameEngine(disable_logging=True)

    def read(self, file_path: str):
        for reversi_data_path in glob.glob(file_path):
            logger.info(f"Reading {reversi_data_path}")
            for game_data in tqdm.tqdm(parse_wtb_file(reversi_data_path)):
                try:
                    instances = self._convert_moves(game_data["moves"])
                except InvalidPositionError:
                    continue
                yield from instances

    def _convert_moves(self, moves):
        instances = []
        self._game_engine.reset()
        for move in moves:
            position = Position.from_move(move)
            move_index = position_to_index(position, size=self._game_engine.board.size)

            instance = {
                "board_feature": self.feature_extractor(self._game_engine.board, self._game_engine.current_color),
                "move": move_index,
            }
            instances.append(instance)

            position = Position.from_move(move)
            self._game_engine.execute_move(position)
        return instances

    def generate_batches(self, file_path: str, batch_size: int, shuffle: bool) -> DataLoader:
        for reversi_data_path in glob.glob(file_path):
            dataset = [instance for instance in self.read(reversi_data_path)]
            data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
            for batch in data_loader:
                yield batch
