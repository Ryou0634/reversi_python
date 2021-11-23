import click
import json
import tqdm
from collections import Counter


from registrable import import_submodules
from reversi.players import Player
from reversi.board import ReversiBoard, Color
from reversi.game_engine import ReversiGameEngine


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
@click.option("--num-games", type=int, default=100)
def play(black_player: str, white_player: str, board_type: str, num_games: int):
    import_submodules("reversi")
    import_submodules("search_algorithm")

    counter = Counter()

    players = {Color.BLACK: load_player(black_player, Color.BLACK), Color.WHITE: load_player(white_player, Color.WHITE)}
    game_engine = ReversiGameEngine(ReversiBoard.by_name(board_type)())

    for _ in tqdm.tqdm(range(num_games)):
        game_engine.reset()
        is_terminal = False
        while not is_terminal:
            legal_positions = game_engine.board.get_legal_positions(game_engine.current_color)
            position = players[game_engine.current_color].choose_position(game_engine.board, legal_positions)
            is_terminal = game_engine.execute_move(position)
        result = game_engine.summarize_result()
        counter[result.winner] += 1
    print(counter)


if __name__ == "__main__":
    play()
