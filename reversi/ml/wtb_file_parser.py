from typing import List
import ctypes


class WthorHeader(ctypes.Structure):
    """
    header:
        date: 4
        num_games: 4
        misc_data: 4
        padding 4
    """

    _fields_ = (
        ("century", ctypes.c_byte),
        ("year", ctypes.c_byte),
        ("month", ctypes.c_byte),
        ("day", ctypes.c_byte),
        ("num_games", ctypes.c_int32),
        ("num_contest", ctypes.c_int16),
        ("year2", ctypes.c_int16),
        ("padding", ctypes.c_int32),
    )


class WthorStructure(ctypes.Structure):
    """
    contest index: 2
    black id: 2
    white id: 2
    black result: 1
    maximum result: 1
    move seq: 1*60
    """

    _fields_ = (
        ("contest_index", ctypes.c_int16),
        ("black_id", ctypes.c_int16),
        ("white_id", ctypes.c_int16),
        ("black_result", ctypes.c_int8),
        ("maximum_result", ctypes.c_int8),
        ("moves", ctypes.c_int8 * 60),
    )


class InvalidDataException(Exception):
    pass


def parse_moves(moves: List[ctypes.c_int8]) -> List[str]:
    def _parse_move(move: ctypes.c_int8) -> str:
        x = move % 10
        y = move // 10
        if not 1 <= x <= 8 and 1 <= y <= 8:
            raise InvalidDataException()

        x_begin = "a"
        y_begin = "1"

        return "{}{}".format(
            *[chr(ord(beg) + (z - 1)) for z, beg in zip([x, y], [x_begin, y_begin])]
        )

    num_valid = 0
    for move in moves:
        # invalid data is represented by 0
        if move == 0:
            break
        num_valid += 1
    return_list = []
    for move in moves[:num_valid]:
        try:
            parsed_move = _parse_move(move)
            return_list.append(parsed_move)
        except InvalidDataException:
            continue
    return return_list


def parse_wtb_file(filename: str):
    """A short example for parsing wthor files.

    ffo_wthor/ data is found at: http://www.ffothello.org/informatique/la-base-wthor/
    ggs_wthor/ data is found at: https://www.skatgame.net/mburo/ggs/game-archive/Othello/ (ggf files)
    A ggf-to-wthor converter is at https://www.skatgame.net/mburo/ggs/ .
    Wthor structure is described in http;//hp.vector.co.jp/authors/VA015468/platina/algo/append_a.html .
    """
    h = WthorHeader()
    with open(filename, "rb") as f:
        f.readinto(h)
        # print(h.century, h.year, h.month, h.day, h.num_games, h.num_contest, h.year2)
        for i in range(h.num_games):
            t = WthorStructure()
            f.readinto(t)
            # print(t.contest_index, t.black_id, t.white_id)
            moves = parse_moves(t.moves)
            if moves is None:
                continue
            result = t.black_result, len(moves) + 4 - t.black_result
            best = t.maximum_result, len(moves) + 4 - t.maximum_result
            yield {"moves": moves, "result": result, "best": best}
