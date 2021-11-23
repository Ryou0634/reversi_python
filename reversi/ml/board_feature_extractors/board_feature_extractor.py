import numpy as np
from abc import ABCMeta, abstractmethod

from registrable import Registrable
from reversi.board import ReversiBoard, Color


class BoardFeatureExtractor(Registrable, metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, board: ReversiBoard, current_color: Color) -> np.ndarray:
        raise NotImplementedError
