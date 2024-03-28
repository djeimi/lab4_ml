from dataclasses import dataclass
from enum import auto
from enum import Enum
from typing import Dict
from typing import Type

import numpy as np


@dataclass
class LearningRate:
    lambda_: float = 1e-3
    s0: float = 1
    p: float = 0.5

    iteration: int = 0

    def __call__(self):
        self.iteration += 1
        return self.lambda_ * (self.s0 / (self.s0 + self.iteration)) ** self.p


class LossFunction(Enum):
    MSE = auto()
    MAE = auto()
    LogCosh = auto()
    Huber = auto()


class BaseDescent:
    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        self.w: np.ndarray = np.random.rand(dimension)
        self.lr: LearningRate = LearningRate(lambda_=lambda_)
        self.loss_function: LossFunction = loss_function

    def step(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.update_weights(self.calc_gradient(x, y))

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        pass

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        pass

    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        return np.mean(np.power((self.predict(x) - y), 2))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return x @ self.w


class VanillaGradientDescent(BaseDescent):
    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        step_size = self.lr()
        self.w -= step_size * gradient
        return -step_size * gradient

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        errors = self.predict(x) - y
        gradient = 2 * x.T @ errors / x.shape[0]
        return gradient


class StochasticDescent(VanillaGradientDescent):
    def __init__(self, dimension: int, lambda_: float = 1e-3, batch_size: int = 50,
                 loss_function: LossFunction = LossFunction.MSE):

        super().__init__(dimension, lambda_, loss_function)
        self.batch_size = batch_size

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        index = np.random.randint(y.shape[0], size=self.batch_size)
        return super().calc_gradient(x[index, :], y[index])


class MomentumDescent(VanillaGradientDescent):
    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        super().__init__(dimension, lambda_, loss_function)
        self.alpha: float = 0.9

        self.h: np.ndarray = np.zeros(dimension)

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        self.h = self.alpha * self.h + self.lr() * gradient
        self.w -= self.h
        return -self.h


class Adam(VanillaGradientDescent):
    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        super().__init__(dimension, lambda_, loss_function)
        self.eps: float = 1e-8

        self.m: np.ndarray = np.zeros(dimension)
        self.v: np.ndarray = np.zeros(dimension)

        self.beta_1: float = 0.9
        self.beta_2: float = 0.999

        self.iteration: int = 0

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        self.iteration += 1

        self.m = self.beta_1 * self.m + (1 - self.beta_1) * gradient
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * gradient ** 2

        m_corrected = self.m / (1 - self.beta_1 ** self.iteration)
        v_corrected = self.v / (1 - self.beta_2 ** self.iteration)

        diff_w = self.lr() * m_corrected / (np.sqrt(v_corrected + self.eps))

        self.w -= diff_w
        return -diff_w


class BaseDescentReg(BaseDescent):
    def __init__(self, *args, mu: float = 0, **kwargs):
        super().__init__(*args, **kwargs)

        self.mu = mu

    def step(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        gradient = self.calc_gradient(x, y)
        return self.update_weights(gradient)

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        pass

    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        return np.mean(np.power((self.predict(x) - y), 2)) + self.mu / 2 * np.sum(np.power(self.w, 2))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return x @ self.w

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        l2_gradient: np.ndarray = np.zeros_like(x.shape[1])  

        return super().calc_gradient(x, y) + l2_gradient * self.mu


class VanillaGradientDescentReg(BaseDescentReg, VanillaGradientDescent):
    """
    Класс полного градиентного спуска с регуляризацией.
    """


class StochasticDescentReg(BaseDescentReg, StochasticDescent):
    """
    Класс стохастического градиентного спуска с регуляризацией.
    """


class MomentumDescentReg(BaseDescentReg, MomentumDescent):
    """
    Класс градиентного спуска с моментом и регуляризацией.
    """


class AdamReg(BaseDescentReg, Adam):
    """
    Класс адаптивного градиентного алгоритма с регуляризацией (AdamReg).
    """


def get_descent(descent_config: dict) -> BaseDescent:
    """
    Создает экземпляр класса градиентного спуска на основе предоставленной конфигурации.

    Параметры
    ----------
    descent_config : dict
        Словарь конфигурации для выбора и настройки класса градиентного спуска. Должен содержать ключи:
        - 'descent_name': строка, название метода спуска ('full', 'stochastic', 'momentum', 'adam').
        - 'regularized': булево значение, указывает на необходимость использования регуляризации.
        - 'kwargs': словарь дополнительных аргументов, передаваемых в конструктор класса спуска.

    Возвращает
    -------
    BaseDescent
        Экземпляр класса, реализующего выбранный метод градиентного спуска.

    Исключения
    ----------
    ValueError
        Вызывается, если указано неправильное имя метода спуска.

    Примеры
    --------
    >>> descent_config = {
    ...     'descent_name': 'full',
    ...     'regularized': True,
    ...     'kwargs': {'dimension': 10, 'lambda_': 0.01, 'mu': 0.1}
    ... }
    >>> descent = get_descent(descent_config)
    >>> isinstance(descent, BaseDescent)
    True
    """
    descent_name = descent_config.get('descent_name', 'full')
    regularized = descent_config.get('regularized', False)

    descent_mapping: Dict[str, Type[BaseDescent]] = {
        'full': VanillaGradientDescent if not regularized else VanillaGradientDescentReg,
        'stochastic': StochasticDescent if not regularized else StochasticDescentReg,
        'momentum': MomentumDescent if not regularized else MomentumDescentReg,
        'adam': Adam if not regularized else AdamReg
    }

    if descent_name not in descent_mapping:
        raise ValueError(f'Incorrect descent name, use one of these: {descent_mapping.keys()}')

    descent_class = descent_mapping[descent_name]

    return descent_class(**descent_config.get('kwargs', {}))
