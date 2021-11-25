from typing import List, Dict
import random
import time

from .search_algorithm import SearchAlgorithm
from .tree_node import TreeNode, Action, NodeEvaluator
from .time_limit import set_start_time, quit_when_time_over

import logging

logger = logging.getLogger(__name__)


@SearchAlgorithm.register("monte_carlo")
class MonteCarloSearch(SearchAlgorithm):
    def __init__(self, node_evaluator: NodeEvaluator, max_time: float = None, max_num_playouts: int = 10000):
        self.node_evaluator = node_evaluator
        self.max_time = max_time
        self.max_num_playouts = max_num_playouts
        self._start_time = 0

    @set_start_time
    def search_best_action(self, current_node: TreeNode) -> Action:
        self._start_time = time.time()
        valid_actions = current_node.get_valid_actions()
        # randomly choice an action in case that cannot perform search within the time

        playout_scores_for_actions: Dict[Action, List[float]] = {action: [] for action in valid_actions}

        try:
            for num_playouts in range(self.max_num_playouts):
                action = random.choice(valid_actions)
                score = self.playout(current_node.get_next_node(action))
                playout_scores_for_actions[action].append(score)
        except TimeoutError:
            logger.info(f"Run out of time.")

        logger.info(f"Number of playouts: {num_playouts}")

        average_score_for_actions = {
            action: sum(scores) / len(scores) for action, scores in playout_scores_for_actions.items() if scores
        }
        best_action, best_score = max(average_score_for_actions.items(), key=lambda x: x[1])

        logger.info(f"Evaluated score: {best_score}")

        return best_action

    @quit_when_time_over
    def playout(self, start_node: TreeNode) -> float:
        current_node = start_node
        while not current_node.is_terminal:
            actions = current_node.get_valid_actions()
            action = random.choice(actions)
            current_node = current_node.get_next_node(action)

        return self.node_evaluator(current_node)
