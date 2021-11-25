from typing import Any
import random
import math
from .time_limit import set_start_time, quit_when_time_over
from .tree_node import Action, TreeNode, NodeEvaluator, ScoredAction
from .search_algorithm import SearchAlgorithm
import logging

logger = logging.getLogger(__name__)


@SearchAlgorithm.register("min_max")
class MinMaxSearch(SearchAlgorithm):
    def __init__(self, node_evaluator: NodeEvaluator, max_depth: int = 100, max_time: float = None):
        self.node_evaluator = node_evaluator
        self.max_depth = max_depth
        self.max_time = max_time
        self._start_time = 0

    @set_start_time
    def search_best_action(self, current_node: TreeNode) -> Action:
        # randomly choice an action in case that cannot perform search within the time
        action = random.choice(current_node.get_valid_actions())
        best_scored_action = ScoredAction(action, -math.inf)

        # iterative depth-first search
        try:
            max_depth = 1
            while max_depth <= self.max_depth:
                logger.info(f"Searching depth {max_depth}...")
                best_scored_action = self._search_best_action_by_depth(current_node, max_depth=max_depth)
                if abs(best_scored_action.score) == math.inf:
                    break
                max_depth += 1
        except TimeoutError:
            logger.info(f"Run out of time.")

        logger.info(f"Evaluated score: {best_scored_action.score}")

        return best_scored_action.action

    def _search_best_action_by_depth(self, current_node: TreeNode, max_depth: int) -> ScoredAction:
        assert max_depth > 0
        best_scored_action = ScoredAction(None, -math.inf)
        for a in current_node.get_valid_actions():
            score = self.evaluate_move(a, current_node, current_depth=0, max_depth=max_depth)
            best_scored_action = max(best_scored_action, ScoredAction(a, score), key=lambda x: x.score)
        return best_scored_action

    @quit_when_time_over
    def evaluate_move(self, action: Any, current_node: TreeNode, current_depth: int, max_depth: int) -> float:
        assert current_depth < max_depth, f"{current_depth} < {max_depth}"
        assert not current_node.is_terminal

        next_node = current_node.get_next_node(action)
        next_depth = current_depth + 1

        if next_depth == max_depth or next_node.is_terminal:
            best_score = self.node_evaluator(next_node)
        else:
            if next_node.is_opponent_turn:
                best_score = math.inf
                eval_func = min
            else:
                best_score = -math.inf
                eval_func = max
            for next_possible_action in next_node.get_valid_actions():
                score = self.evaluate_move(next_possible_action, next_node, next_depth, max_depth)
                best_score = eval_func(best_score, score)
        return best_score
