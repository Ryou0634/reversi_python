from abc import abstractmethod
from registrable import Registrable
from .tree_node import TreeNode, Action


class SearchAlgorithm(Registrable):
    @abstractmethod
    def search_best_action(self, current_node: TreeNode) -> Action:
        raise NotImplementedError
