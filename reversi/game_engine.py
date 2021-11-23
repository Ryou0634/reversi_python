from typing import Optional, List, Tuple, Set, NamedTuple, Dict
import copy

from reversi.board import ReversiBoard, Color, Position
from reversi.board.bit_board import BitBoard

import logging

logger = logging.getLogger(__name__)


def dump_board_string(board: ReversiBoard, legal_positions_of: Color = None) -> str:
    legal_positions: Set[Position] = set()
    if legal_positions_of is not None:
        legal_positions.update(board.get_legal_positions(legal_positions_of))

    axis_names = [str(Position(i, i)) for i in range(board.size)]
    return_str = "  " + " ".join([name[0] for name in axis_names])
    for x in range(board.size):
        return_str += "\n" + axis_names[x][1]
        for y in range(board.size):
            position = Position(x, y)
            color = board.get_color(position)
            if color == Color.BLACK:
                cell_state = "●"
            elif color == Color.WHITE:
                cell_state = "○"
            else:
                cell_state = "-"

            if position in legal_positions:
                cell_state = "*"

            return_str += " " + str(cell_state)
    return "\n" + return_str


class ReversiResult(NamedTuple):
    num_disks: Dict[Color, int]
    winner: Optional[Color]
    message: str


class ReversiGameEngine:
    def __init__(self, board: ReversiBoard = None, disable_logging: bool = False):
        self.board = board or BitBoard()

        self.current_color = None
        self.snapshots: List[Tuple[ReversiBoard, Color]] = []

        self.logging_func = logger.info
        if disable_logging:
            self.logging_func = logger.debug

    def reset(self):
        self.board.reset()
        self.current_color = Color.BLACK
        self.snapshots = []

        self.logging_func(dump_board_string(self.board))
        self.logging_func(f"Current Player: {self.current_color}")

    def execute_move(self, position: Position) -> bool:
        self.snapshots.append((copy.deepcopy(self.board), self.current_color))
        self.board.place(position, self.current_color)

        self.logging_func(f"{self.current_color} placed on {position}")

        is_terminal = False
        if len(self.board.get_legal_positions(self.current_color.opponent)) > 0:
            self.current_color = self.current_color.opponent
            self.logging_func(f"Current Player: {self.current_color}")
        elif len(self.board.get_legal_positions(self.current_color)) > 0:
            self.logging_func(f"{self.current_color.opponent} skipped.")
            pass
        else:
            is_terminal = True

        self.logging_func(
            dump_board_string(self.board, legal_positions_of=self.current_color)
        )
        return is_terminal

    def restore_from_latest_snapshot(self):
        if len(self.snapshots) == 0:
            return
        self.board, self.current_color = self.snapshots.pop()

        self.logging_func("Restored from the latest snapshot.")
        self.logging_func(
            dump_board_string(self.board, legal_positions_of=self.current_color)
        )

    def summarize_result(self) -> ReversiResult:

        num_disks = {
            Color.BLACK: self.board.get_num_disks(Color.BLACK),
            Color.WHITE: self.board.get_num_disks(Color.WHITE),
        }

        if num_disks[Color.BLACK] > num_disks[Color.WHITE]:
            message = "Black Wins!"
            winner = Color.BLACK
        elif num_disks[Color.BLACK] < num_disks[Color.WHITE]:
            message = "White Wins!"
            winner = Color.WHITE
        else:
            message = "Draw."
            winner = None

        result = ReversiResult(num_disks, winner, message)
        self.logging_func(result)
        return result
