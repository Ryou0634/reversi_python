from typing import Optional
import time

from reversi.board import Color
from .game_interface import GameInterface


@GameInterface.register("cui")
class CuiGameInterface(GameInterface):
    def play(self, interval: int = 1.0) -> Optional[Color]:
        self.game_engine.reset()

        is_terminal = False
        while not is_terminal:
            legal_positions = self.game_engine.board.get_legal_positions(self.game_engine.current_color)
            position = self.players[self.game_engine.current_color].choose_position(
                self.game_engine.board, legal_positions
            )
            is_terminal = self.game_engine.execute_move(position)
            time.sleep(interval)

        result = self.game_engine.summarize_result()
        return result.winner
