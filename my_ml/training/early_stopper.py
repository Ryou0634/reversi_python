import numpy as np
import logging

logger = logging.getLogger(__name__)


class EarlyStopper:
    def __init__(self, patience: int = 5, delta: float = 0, reverse: bool = False):

        self.patience = patience
        self.counter = 0

        self.sign = -1 if reverse else 1

        self.delta = delta * self.sign
        self.current_score = -np.inf * self.sign
        self.best_score = self.current_score

        self.current_is_best = True

    def update(self, score: float):

        if self.sign * score < self.sign * self.best_score + self.delta:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            self.current_is_best = False

        else:
            self.best_score = score
            self.counter = 0
            self.current_is_best = True

        self.current_score = score

    def to_stop(self) -> bool:
        return self.counter >= self.patience
