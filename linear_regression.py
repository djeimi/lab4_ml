from __future__ import annotations

from typing import List

import numpy as np

from descents import BaseDescent
from descents import get_descent


class LinearRegression:
    def __init__(self, descent_config: dict, tolerance: float = 1e-4, max_iter: int = 300):
        self.descent: BaseDescent = get_descent(descent_config)

        self.tolerance: float = tolerance
        self.max_iter: int = max_iter

        self.loss_history: List[float] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> LinearRegression:
        prev_weights = self.descent.w.copy()
        self.loss_history.append(self.calc_loss(x, y))

        for i in range(self.max_iter):
            self.descent.step(x, y)
            curr_weights = self.descent.w

            if np.linalg.norm(prev_weights - curr_weights) < self.tolerance:
                self.loss_history.append(self.calc_loss(x, y))
                break

            if np.isnan(curr_weights).any():
                raise ValueError("Вектор весов содержит NaN")

            prev_weights = curr_weights
            self.loss_history.append(self.calc_loss(x, y))

        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.descent.predict(x)

    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        return self.descent.calc_loss(x, y)
