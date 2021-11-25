from search_algorithm.min_max_search import NodeEvaluator

from reversi.players.search_players.search_player import ReversiSearchNode


@NodeEvaluator.register("reversi_win_lose")
class WinLoseEvaluator(NodeEvaluator):
    def __call__(self, node: ReversiSearchNode) -> float:
        assert node.is_terminal

        board = node.board
        num_player_disks = board.get_num_disks(node.playing_color)
        num_opponent_disks = board.get_num_disks(node.playing_color.opponent)

        if num_player_disks > num_opponent_disks:
            return 1.0
        elif num_player_disks < num_opponent_disks:
            return 0
        else:
            return 0.5
