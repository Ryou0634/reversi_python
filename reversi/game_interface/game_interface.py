from typing import Optional

from registrable import Registrable
from reversi.players import Player
from reversi.board import ReversiBoard, Color
from reversi.game_engine import ReversiGameEngine

import logging

logger = logging.getLogger(__name__)


class GameInterface(Registrable):
    def __init__(self, board: ReversiBoard, black_player: Player, white_player: Player):
        assert black_player.color == Color.BLACK
        assert white_player.color == Color.WHITE
        self.players = {Color.BLACK: black_player, Color.WHITE: white_player}
        self.game_engine = ReversiGameEngine(board)

    def play(self) -> Optional[Color]:
        raise NotImplementedError()
