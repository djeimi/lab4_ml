import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import linear_regression

from descents import get_descent
from linear_regression import LinearRegression

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

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

#plt.show()
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

#plt.show()

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
max_iter = 100
tolerance = 0

descent_config = {
    'descent_name': 'stochastic',
    'kwargs': {
        'dimension': dimension,
        'batch_size': 10
    }
}

regression = linear_regression.LinearRegression(
    descent_config=descent_config,
    tolerance=tolerance,
    max_iter=max_iter
)

# X_train = X_train.toarray()
# y_train = y_train.values
# y_train = np.reshape(y_train, (-1, 1))

n = 5000  # количество первых значений
X_train_subset = X_train[:n]
y_train_subset = y_train[:n]

regression.fit(X_train_subset, y_train_subset)

plt.plot(regression.loss_history)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()
