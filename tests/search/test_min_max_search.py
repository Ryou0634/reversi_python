from typing import List
import pytest

from search_algorithm.min_max_search import MinMaxSearchNode, SearchNodeEvaluator, MinMaxSearch


class FakeSearchNode(MinMaxSearchNode):
    def __init__(self, state_value: int, is_opponent_turn: bool):
        self.state_value = state_value
        self._is_opponent_turn = is_opponent_turn

    def get_valid_actions(self) -> List[int]:
        return [-1, 0, 1]

    def get_next_node(self, action: int) -> "FakeSearchNode":
        return FakeSearchNode(self.state_value + action, is_opponent_turn=not self.is_opponent_turn)

    @property
    def is_opponent_turn(self) -> bool:
        return self._is_opponent_turn

    @property
    def is_terminal(self) -> bool:
        return self.state_value == 3


class FakeSearchNodeEvaluator(SearchNodeEvaluator):
    def evaluate(self, node: FakeSearchNode):
        return node.state_value


@pytest.mark.parametrize(
    "max_depth, action, expected_score",
    [[1, -1, -1], [1, 1, 1], [1, 0, 0], [2, -1, -2], [2, 1, 0], [2, 0, -1], [3, -1, -1], [3, 1, 1], [3, 0, 0],],
)
def test_if_evaluate_move_returns_correct_score(max_depth: int, action: int, expected_score: int):
    min_max_search = MinMaxSearch(node_evaluator=FakeSearchNodeEvaluator(), max_depth=3)

    current_node = FakeSearchNode(0, is_opponent_turn=False)

    score = min_max_search.evaluate_move(action=action, current_node=current_node, current_depth=0, max_depth=max_depth)
    assert score == expected_score


@pytest.mark.parametrize(
    "max_depth, action, expected_score", [[2, 1, 3], [3, 1, 3],],
)
def test_if_evaluate_move_considers_terminal_state(max_depth: int, action: int, expected_score: int):
    min_max_search = MinMaxSearch(node_evaluator=FakeSearchNodeEvaluator(), max_depth=3)

    current_node = FakeSearchNode(2, is_opponent_turn=False)

    score = min_max_search.evaluate_move(action=action, current_node=current_node, current_depth=0, max_depth=max_depth)
    assert score == expected_score


def test_if_search_returns_best_action():
    min_max_search = MinMaxSearch(node_evaluator=FakeSearchNodeEvaluator(), max_depth=3)

    current_node = FakeSearchNode(0, is_opponent_turn=False)
    assert min_max_search.search_best_action(current_node=current_node) == 1
