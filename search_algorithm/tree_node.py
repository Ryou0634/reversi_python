from typing import List, Any, NewType, NamedTuple
from registrable import Registrable
import logging

logger = logging.getLogger(__name__)


Action = NewType("Action", Any)


class ScoredAction(NamedTuple):
    action: Action
    score: float


class TreeNode(Registrable):
    def get_valid_actions(self) -> List[Action]:
        raise NotImplementedError

    def get_next_node(self, action: Action) -> "TreeNode":
        raise NotImplementedError

    @property
    def is_opponent_turn(self) -> bool:
        raise NotImplementedError

    @property
    def is_terminal(self) -> bool:
        raise NotImplementedError


class NodeEvaluator(Registrable):
    def __call__(self, node: TreeNode) -> float:
        raise NotImplementedError
