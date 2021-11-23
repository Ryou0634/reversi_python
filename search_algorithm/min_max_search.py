from typing import List, Any, NewType, NamedTuple
import time
import random
import math
from registrable import Registrable, FromParams
from .time_limit import time_limit

import logging

logger = logging.getLogger(__name__)


Action = NewType("Action", Any)


class ScoredAction(NamedTuple):
    action: Action
    score: float


class MinMaxSearchNode(Registrable):
    def get_valid_actions(self) -> List[Action]:
        raise NotImplementedError

    def get_next_node(self, action: Action) -> "MinMaxSearchNode":
        raise NotImplementedError

    @property
    def is_opponent_turn(self) -> bool:
        raise NotImplementedError

    @property
    def is_terminal(self) -> bool:
        raise NotImplementedError


class SearchNodeEvaluator(Registrable):
    def evaluate(self, node: MinMaxSearchNode) -> float:
        raise NotImplementedError


class MinMaxSearch(FromParams):
    def __init__(self, node_evaluator: SearchNodeEvaluator, max_depth: int = 10, max_time: float = None):
        self.node_evaluator = node_evaluator
        self.max_depth = max_depth
        self.max_time = max_time
        self._start_time = 0

    def search_best_action(self, current_node: MinMaxSearchNode) -> Action:

        if self.max_time is not None:
            self._start_time = time.time()
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
        else:
            best_scored_action = self._search_best_action_by_depth(current_node, max_depth=self.max_depth)

        logger.info(f"Evaluated score: {best_scored_action.score}")

        return best_scored_action.action

    def _search_best_action_by_depth(self, current_node: MinMaxSearchNode, max_depth: int) -> ScoredAction:
        assert max_depth > 0
        best_scored_action = ScoredAction(None, -math.inf)
        for a in current_node.get_valid_actions():
            score = self.evaluate_move(a, current_node, current_depth=0, max_depth=max_depth)
            best_scored_action = max(best_scored_action, ScoredAction(a, score), key=lambda x: x.score)
        return best_scored_action

    @time_limit
    def evaluate_move(self, action: Any, current_node: MinMaxSearchNode, current_depth: int, max_depth: int) -> float:
        assert current_depth < max_depth, f"{current_depth} < {max_depth}"
        assert not current_node.is_terminal

        next_node = current_node.get_next_node(action)
        next_depth = current_depth + 1

        if next_depth == max_depth or next_node.is_terminal:
            best_score = self.node_evaluator.evaluate(next_node)
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
