import numpy as np
import linear_regression
import matplotlib.pyplot as plt

num_objects = 100
dimension = 5

x = np.random.rand(num_objects, dimension)
y = np.random.rand(num_objects)

print(x.dtype)
print(x.shape)
print(y.dtype)
print(y.shape)

max_iter = 100
tolerance = 0

descent_config = {
    'descent_name': 'full',
    'kwargs': {
        'dimension': dimension
    }
}

regression = linear_regression.LinearRegression(
    descent_config=descent_config,
    tolerance=tolerance,
    max_iter=max_iter
)

regression.fit(x, y)

assert len(regression.loss_history) == max_iter + 1, 'Loss history failed'

plt.plot(regression.loss_history)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()