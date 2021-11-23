from typing import List

from reversi.board import Position
from .player import Player, ReversiBoard

import logging

logger = logging.getLogger(__name__)


@Player.register("human")
class HumanPlayer(Player):
    def choose_position(self, current_board: ReversiBoard, legal_positions: List[Position]) -> Position:

        position = None
        while position is None:
            input_string = input("Choose a position to place a disk: ").strip()
            try:
                position = next(p for p in legal_positions if str(p) == input_string)
            except StopIteration:
                logger.info("Illegal position. Choose again.")
                logger.info(f"Legal positions: {[str(p) for p in legal_positions]}.")
        return position
