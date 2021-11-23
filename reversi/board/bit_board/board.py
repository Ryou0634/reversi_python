from typing import Dict, List, NewType, Optional

from reversi.board.color import Color
from reversi.board.position import Position
from reversi.board.board import ReversiBoard
from reversi.board.exceptions import InvalidPositionError


Bits = NewType("Bits", int)


@ReversiBoard.register("bit")
class BitBoard(ReversiBoard):
    def __init__(self, size: int = 8):
        super().__init__(size=size)

        assert self.size == 8, "BitBoard only supports the standard size"

        self.board: Dict[Color, Bits] = {
            Color.BLACK: 0x0000000000000000,
            Color.WHITE: 0x0000000000000000,
        }

        self.reset()

    def _position_to_bits(self, position: Position) -> Bits:
        max_index = self.size ** 2 - 1
        position_index = max_index - ((position.x * self.size) + position.y)
        position_bits: Bits = 1 << position_index
        return position_bits

    def get_color(self, position: Position) -> Optional[Color]:
        position_bits = self._position_to_bits(position)

        for color in list(Color):
            if self.board[color] & position_bits:
                return color
        return None

    def get_legal_positions(self, color: Color) -> List[Position]:
        valid_places = self._get_valid_places(color)

        mask = 1
        valid_actions = []
        for x in reversed(range(self.size)):
            for y in reversed(range(self.size)):
                if bool(mask & valid_places):
                    valid_actions.append(Position(x, y))
                mask <<= 1
        return valid_actions

    def _place(self, position: Position, color: Color):

        position_bits = self._position_to_bits(position)

        reversed_place = self._get_reversed_places(position_bits, color)

        if reversed_place == 0:
            raise InvalidPositionError(position)

        self.board[color] ^= reversed_place ^ position_bits
        self.board[color.opponent] ^= reversed_place

    def get_num_disks(self, color: Color) -> int:

        board = self.board[color]
        count = 0
        while board:
            board &= board - 1
            count += 1
        return count

    def reset(self):
        # we assign a number for each cell from the right lower corner
        # ---------------------
        # 63 ...
        #         ...
        #             ... 2 1 0
        # ---------------------

        self.board = {Color.WHITE: 0x0000001008000000, Color.BLACK: 0x0000000810000000}

    def _get_valid_places(self, color: Color) -> Bits:
        """Generate legal board."""
        player_places = self.board[color]
        opponent_places = self.board[color.opponent]
        blank_places = ~(player_places | opponent_places)

        """
        This mask looks like this.
        × ◯ ◯ × 
        × ◯ ◯ ×
        × ◯ ◯ ×
        × ◯ ◯ ×
        """
        vertically_masked = opponent_places & 0x7E7E7E7E7E7E7E7E
        legal = self._get_valid_left(player_places, vertically_masked, blank_places, 1)  # ←
        legal |= self._get_valid_right(player_places, vertically_masked, blank_places, 1)  # →

        """
        × × × × 
        ◯ ◯ ◯ ◯
        ◯ ◯ ◯ ◯
        × × × ×
        """
        horizontally_masked = opponent_places & 0x00FFFFFFFFFFFF00
        legal |= self._get_valid_left(player_places, horizontally_masked, blank_places, 8)  # ↑
        legal |= self._get_valid_right(player_places, horizontally_masked, blank_places, 8)  # ↓

        """
        × × × × 
        × ◯ ◯ ×
        × ◯ ◯ ×
        × × × ×
        """
        around_masked = opponent_places & 0x007E7E7E7E7E7E00
        legal |= self._get_valid_left(player_places, around_masked, blank_places, 7)  # ↗
        legal |= self._get_valid_left(player_places, around_masked, blank_places, 9)  # ↖
        legal |= self._get_valid_right(player_places, around_masked, blank_places, 7)  # ↙
        legal |= self._get_valid_right(player_places, around_masked, blank_places, 9)  # ↘
        return legal

    @staticmethod
    def _get_valid_left(player_places: Bits, masked_opponent_places: Bits, blank_places: Bits, direction: int,) -> Bits:
        """Direction << dir exploring."""
        tmp = masked_opponent_places & (player_places << direction)
        tmp |= masked_opponent_places & (tmp << direction)
        tmp |= masked_opponent_places & (tmp << direction)
        tmp |= masked_opponent_places & (tmp << direction)
        tmp |= masked_opponent_places & (tmp << direction)
        tmp |= masked_opponent_places & (tmp << direction)
        valid_places = blank_places & (tmp << direction)
        return valid_places

    @staticmethod
    def _get_valid_right(
        player_places: Bits, masked_opponent_places: Bits, blank_places: Bits, direction: int,
    ) -> Bits:
        """Direction >> dir exploring."""
        tmp = masked_opponent_places & (player_places >> direction)
        tmp |= masked_opponent_places & (tmp >> direction)
        tmp |= masked_opponent_places & (tmp >> direction)
        tmp |= masked_opponent_places & (tmp >> direction)
        tmp |= masked_opponent_places & (tmp >> direction)
        tmp |= masked_opponent_places & (tmp >> direction)
        valid_places = blank_places & (tmp >> direction)
        return valid_places

    def _get_reversed_places(self, position_bits, color: Color) -> Bits:
        """Return get_reversed_places site board."""
        player_places = self.board[color]
        opponent_places = self.board[color.opponent]

        blank_h = ~(player_places | opponent_places & 0x7E7E7E7E7E7E7E7E)
        rev = self._get_reversed_left(player_places, blank_h, position_bits, 1)
        rev |= self._get_reversed_right(player_places, blank_h, position_bits, 1)

        blank_v = ~(player_places | opponent_places & 0x00FFFFFFFFFFFF00)

        rev |= self._get_reversed_left(player_places, blank_v, position_bits, 8)
        rev |= self._get_reversed_right(player_places, blank_v, position_bits, 8)

        blank_a = ~(player_places | opponent_places & 0x007E7E7E7E7E7E00)
        rev |= self._get_reversed_left(player_places, blank_a, position_bits, 7)
        rev |= self._get_reversed_left(player_places, blank_a, position_bits, 9)
        rev |= self._get_reversed_right(player_places, blank_a, position_bits, 7)
        rev |= self._get_reversed_right(player_places, blank_a, position_bits, 9)
        return rev

    def _get_reversed_left(self, player_places, masked_blank_spaces: Bits, site, direction: int) -> Bits:
        """Direction << for self.get_reversed_places()."""
        rev: Bits = 0
        opponent_places = ~(player_places | masked_blank_spaces) & (site << direction)
        if opponent_places:
            for i in range(6):
                opponent_places <<= direction
                if opponent_places & masked_blank_spaces:
                    break
                elif opponent_places & player_places:
                    rev |= opponent_places >> direction
                    break
                else:
                    opponent_places |= opponent_places >> direction
        return rev

    def _get_reversed_right(self, player_places, masked_blank_spaces: Bits, site, direction: int) -> Bits:
        """Direction >> for self.get_reversed_places()."""
        rev: Bits = 0
        opponent_places = ~(player_places | masked_blank_spaces) & (site >> direction)
        if opponent_places:
            for i in range(6):
                opponent_places >>= direction
                if opponent_places & masked_blank_spaces:
                    break
                elif opponent_places & player_places:
                    rev |= opponent_places << direction
                    break
                else:
                    opponent_places |= opponent_places << direction
        return rev


def bit_to_boolean(bitboard: Bits, size: int):
    boolean_board = [[0 for _ in range(size)] for _ in range(size)]
    mask = 1
    for row in reversed(range(size)):
        for col in reversed(range(size)):
            boolean_board[row][col] = bool(mask & bitboard)
            mask <<= 1
    return boolean_board


def bit_to_strings(bits: int, size: int):
    bits_string = bin(bits)[2:].zfill(size * size)
    bits_matrix = ""
    for i in range(size):
        bits_matrix += bits_string[i * size : (i + 1) * size]
        bits_matrix += "\n"
    return bits_matrix
