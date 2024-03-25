#Задания 4-5

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import linear_regression

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

import time

sns.set(style='darkgrid')
data = pd.read_csv('autos.csv')
# print(data.head())

categorical = ['brand', 'model', 'vehicleType', 'gearbox', 'fuelType', 'notRepairedDamage']
numeric = ['powerPS', 'kilometer', 'autoAgeMonths']
target = 'price'

data['log_price'] = np.log(data[target])

fig, axes = plt.subplots(1, 2)
sns.histplot(data=data['log_price'], kde=True, ax=axes[0])
sns.boxplot(data=data['log_price'], ax=axes[1])
#Выбросы имеются

#Обработка выбросов
print("Количество строк до удаления выбросов:", len(data))

Q1 = data['log_price'].quantile(0.25)
Q3 = data['log_price'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = data[(data['log_price'] < lower_bound) | (data['log_price'] > upper_bound)]
data = data.drop(outliers.index)

print("Количество строк после удаления выбросов:", len(data))

# print(data.info())

fig, axs = plt.subplots(nrows=1, ncols=len(categorical), figsize=(20, 5))
for i, col in enumerate(categorical):
    sns.boxplot(data=data, x=col, y='log_price', ax=axs[i])
    axs[i].set_xlabel(col)
    axs[i].set_ylabel('')

fig, axs = plt.subplots(nrows=1, ncols=len(numeric), figsize=(20, 5))
for i, col in enumerate(numeric):
    sns.scatterplot(data=data, x=col, y='log_price', ax=axs[i])
    axs[i].set_xlabel(col)
    axs[i].set_ylabel('')

data['bias'] = 1
other = ['bias']

x = data[categorical + numeric + other]
y = data['log_price']

column_transformer = ColumnTransformer([
    ('ohe', OneHotEncoder(handle_unknown='ignore'), categorical),
    ('scaling', StandardScaler(), numeric)
])

x = column_transformer.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1111, random_state=42)

dimension = x.shape[1]
max_iter = 500
tolerance = 0.2
batch_size = 13

descent_names = ['full', 'stochastic', 'momentum', 'adam']
nrows = 2
ncols = len(descent_names) // nrows if len(descent_names) % nrows == 0 else len(descent_names) // nrows + 1

fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 8))

# for i, descent_name in enumerate(descent_names):
#     ax = axs[i // ncols, i % ncols]
#
#     descent_config = {
#         'descent_name': descent_name,
#         'kwargs': {
#             'dimension': dimension
#         }
#     }
#
#     if descent_name == 'stochastic':
#         descent_config['kwargs']['batch_size'] = batch_size
#
#     regression = linear_regression.LinearRegression(
#         descent_config=descent_config,
#         tolerance=tolerance,
#         max_iter=max_iter
#     )
#
#     n = 5000
#     X_train_subset = X_train[:n]
#     y_train_subset = y_train[:n].to_numpy()
#
#     regression.fit(X_train_subset, y_train_subset)
#
#     ax.plot(regression.loss_history)
#     ax.set_xlabel('Iteration')
#     ax.set_ylabel('Loss')
#     ax.set_title(f'Loss History for {descent_name}')
#
# plt.tight_layout()
#plt.show()
#Каждый метод лучше предыдущего из нашего списка.

#Задание 5.1
alphas = np.logspace(-3, 2, 20)

results = {}


def calculate_r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Вычисляет коэффициент определения R² (r²) между истинными и предсказанными значениями.

    :param y_true: Истинные значения.
    :param y_pred: Предсказанные значения.
    :return: Коэффициент определения R² (r²).
    """
    mean_y_true = np.mean(y_true)
    ss_res = np.sum(np.power(y_true - y_pred, 2))
    ss_tot = np.sum(np.power(y_true - mean_y_true, 2))
    return 1 - (ss_res / ss_tot)
#
# x = data[numeric + other]
# y = data['log_price']
#
# column_transformer = ColumnTransformer([
#     ('scaling', StandardScaler(), numeric)
# ])
#
# x = column_transformer.fit_transform(x)
#
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1111, random_state=42)
#
# dimension = x.shape[1]
#


for i, descent_name in enumerate(descent_names):
    ax = axs[i // ncols, i % ncols]

    descent_config = {
        'descent_name': descent_name,
        'kwargs': {
            'dimension': dimension
        }
    }

    if descent_name == 'stochastic':
        descent_config['kwargs']['batch_size'] = batch_size

    best_alpha = None
    best_loss = float('+inf')
    errors_train = []
    errors_val = []
    iterations = []

    for alpha in alphas:
        descent_config['kwargs']['lambda_'] = alpha

        regression = linear_regression.LinearRegression(
            descent_config=descent_config,
            tolerance=tolerance,
            max_iter=max_iter
        )

        # n = 5000
        # X_train_subset = X_train[:n]
        # y_train_subset = y_train[:n].to_numpy()
        # X_val_subset = X_val[:n]
        # y_val_subset = y_val[:n].to_numpy()

        regression.fit(X_train, y_train.to_numpy())

        r2_train = calculate_r2_score(y_train, regression.predict(X_train))
        r2_val = calculate_r2_score(y_val, regression.predict(X_val))

        error_train = regression.descent.calc_loss(X_train, y_train)
        error_val = regression.descent.calc_loss(X_val, y_val)

        errors_train.append(error_train)
        errors_val.append(error_val)
        iterations.append(regression.n_iter_)
        print(f'Error_val: {error_val}')

        if error_val < best_loss:
            best_loss = error_val
            best_alpha = alpha

    results[descent_name] = {
        'best_alpha': best_alpha,
        'best_r2_val': best_loss,
        'errors_train': errors_train,
        'errors_val': errors_val,
        'iterations': iterations
    }

    print(f"Лучшее значение Λ для {descent_name}: {best_alpha}, ошибка на валидационном наборе: {best_loss}")

fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 8))

for i, (descent_name, data) in enumerate(results.items()):
    row = i // ncols
    col = i % ncols

    ax = axs[row, col] if nrows * ncols > 1 else axs

    ax.plot(data['iterations'], data['errors_train'], label='Тренировочный набор')
    ax.plot(data['iterations'], data['errors_val'], label='Валидационный набор')
    ax.set_title(f'Зависимость ошибки от количества итераций для {descent_name}')
    ax.set_xlabel('Количество итераций')
    ax.set_ylabel('Ошибка')
    ax.legend()

plt.tight_layout()

#Задание 5.2
fig, ax = plt.subplots(figsize=(10, 6))

for descent_name, data in results.items():
    ax.plot(data['iterations'], data['errors_train'], label=descent_name)

ax.set_xlabel('Количество итераций')
ax.set_ylabel('Ошибка')
ax.set_title('Динамика изменения ошибки на обучающей выборке в зависимости от номера итерации')

ax.legend()

plt.tight_layout()
plt.show()

#Задание 6
# batch_sizes = np.arange(5, 500, 10)
# k = 5
#
# avg_times_list = []
# avg_iterations_list = []
#
# for batch_size in batch_sizes:
#     print(f"Batch size: {batch_size}")
#     times = []
#     iterations = []
#
#     for _ in range(k):
#         descent_config = {
#             'descent_name': 'stochastic',
#             'kwargs': {
#                 'dimension': dimension,
#                 'batch_size': batch_size
#             }
#         }
#
#         regression = linear_regression.LinearRegression(
#             descent_config=descent_config,
#             tolerance=tolerance,
#             max_iter=max_iter
#         )
#
#         n = 5000
#         X_train_subset = X_train[:n]
#         y_train_subset = y_train[:n].to_numpy()
#
#         start_time = time.time()
#         regression.fit(X_train_subset, y_train_subset)
#         end_time = time.time()
#
#         times.append(end_time - start_time)
#         iterations.append(len(regression.loss_history))
#
#     avg_time = np.mean(times)
#     avg_iterations = np.mean(iterations)
#     print(f"Average time: {avg_time} seconds")
#     print(f"Average iterations: {avg_iterations}")
#     print()
#
#     avg_times_list.append(avg_time)
#     avg_iterations_list.append(avg_iterations)
#
# plt.figure(figsize=(12, 6))
#
# plt.subplot(1, 2, 1)
# plt.plot(batch_sizes, avg_iterations_list)
# plt.xlabel("Batch size")
# plt.ylabel("Average number of iterations")
#
# plt.subplot(1, 2, 2)
# plt.plot(batch_sizes, avg_times_list)
# plt.xlabel("Batch size")
# plt.ylabel("Average training time (seconds)")
#
# plt.tight_layout()
# plt.show()
