import pytest
from reversi.board import ReversiBoard, Color, Position
from reversi.board.exceptions import InvalidPositionError

from registrable import import_submodules

import_submodules("reversi")


def test_if_color_has_correct_opponent():
    assert Color.WHITE.opponent == Color.BLACK
    assert Color.BLACK.opponent == Color.WHITE


@pytest.fixture
def bit_board():
    return ReversiBoard.by_name("bit")()


@pytest.fixture
def list_board():
    return ReversiBoard.by_name("list")()


@pytest.fixture(params=["list_board", "bit_board"])
def board(request):
    return request.getfixturevalue(request.param)


def test_if_board_has_correct_initial_state(board: ReversiBoard):
    assert board.get_color(Position(3, 3)) == Color.WHITE
    assert board.get_color(Position(3, 4)) == Color.BLACK
    assert board.get_color(Position(4, 3)) == Color.BLACK
    assert board.get_color(Position(4, 4)) == Color.WHITE


def test_if_board_gives_correct_legal_positions(board: ReversiBoard):

    assert set(board.get_legal_positions(Color.WHITE)) == {
        Position(3, 5),
        Position(5, 3),
        Position(2, 4),
        Position(4, 2),
    }

    assert set(board.get_legal_positions(Color.BLACK)) == {
        Position(3, 2),
        Position(2, 3),
        Position(4, 5),
        Position(5, 4),
    }


def test_if_board_raises_invalid_position_error_and_does_not_change(board: ReversiBoard):

    for color in [Color.BLACK, Color.WHITE]:
        for illegal_position in [
            Position(3, 3),
            Position(3, 4),
            Position(4, 3),
            Position(4, 4),
            Position(0, 0),
            Position(7, 7),
        ]:

            original_color = board.get_color(illegal_position)

            with pytest.raises(InvalidPositionError):
                board.place(illegal_position, color)

            assert board.get_color(illegal_position) == original_color


def test_if_board_correctly_flips_disks(board: ReversiBoard):
    board.place(Position(3, 2), Color.BLACK)

    assert board.get_color(Position(3, 2)) == Color.BLACK
    assert board.get_color(Position(3, 3)) == Color.BLACK
    assert board.get_color(Position(3, 4)) == Color.BLACK
    assert board.get_color(Position(4, 3)) == Color.BLACK
    assert board.get_color(Position(4, 4)) == Color.WHITE


def test_if_board_gives_correct_legal_positions_after_place(board: ReversiBoard):
    board.place(Position(3, 2), Color.BLACK)

    assert set(board.get_legal_positions(Color.WHITE)) == {
        Position(2, 2),
        Position(2, 4),
        Position(4, 2),
    }

    assert set(board.get_legal_positions(Color.BLACK)) == {
        Position(4, 5),
        Position(5, 4),
        Position(5, 5),
    }


def test_get_num_disks(board: ReversiBoard):
    assert board.get_num_disks(Color.WHITE) == 2
    assert board.get_num_disks(Color.BLACK) == 2


def test_get_num_disks_after_place(board: ReversiBoard):
    board.place(Position(3, 2), Color.BLACK)

    assert board.get_num_disks(Color.WHITE) == 1
    assert board.get_num_disks(Color.BLACK) == 4
