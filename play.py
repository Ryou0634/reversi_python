import click
import json

from registrable import import_submodules
from reversi.players import Player
from reversi.board import ReversiBoard, Color
from reversi.game_interface import GameInterface
import logging

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def load_player(player_signature: str, color: Color) -> Player:
    try:
        config = json.load(open(player_signature))
        player = Player.from_params(config, color=color)
    except FileNotFoundError:
        player = Player.by_name(player_signature)(color=color)
    return player


@click.command()
@click.argument("black-player", type=str)
@click.argument("white-player", type=str)
@click.option("--board-type", type=str, default="bit")
@click.option("--interface-type", type=str, default="cui")
def play(black_player: str, white_player: str, board_type: str, interface_type: str):
    import_submodules("reversi")
    import_submodules("search_algorithm")
    import_submodules("my_ml")

    game = GameInterface.by_name(interface_type)(
        black_player=load_player(black_player, Color.BLACK),
        white_player=load_player(white_player, Color.WHITE),
        board=ReversiBoard.by_name(board_type)(),
    )

    game.play()


if __name__ == "__main__":
    play()
