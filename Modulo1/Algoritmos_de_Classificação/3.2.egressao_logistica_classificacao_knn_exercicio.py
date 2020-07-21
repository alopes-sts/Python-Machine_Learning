# Exemplo de regressão logitica para classificação usando KNN
# Dataset referente a classificação de um tipo especifico de flor

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

iris = load_iris()
x = pd.DataFrame(iris.data, columns=[iris.feature_names])
y = pd.Series(iris.target)

# somente para estudos, verificar o dataset
# print(x.head())
# print(x.shape)
# impressão do dataset, apenas para visualizar
# print(iris)

# normalização
normalizador = MinMaxScaler(feature_range=(0, 1))
X_norm = normalizador.fit_transform(x)

# definindo os valores que serão testados
valores_K = np.array([3, 5, 7, 9, 11])
calculo_distancia = ['minkowski','chebyshev']
valores_p = np.array([1, 2, 3, 4])
valores_grid = {'n_neighbors':valores_K, 'metric':calculo_distancia, 'p':valores_p}

# criando o modelo
modelo = KNeighborsClassifier()

# criando os grids
gridKNN = GridSearchCV(estimator=modelo, param_grid=valores_grid, cv=5)
gridKNN.fit(X_norm, y)

# imprimindo os melhores parâmetros
print('Melhor acurácia: ', gridKNN.best_score_)
print('Melhor K: ', gridKNN.best_estimator_.n_neighbors)
print('Método distância: ', gridKNN.best_estimator_.metric)
print('Melhor valor p: ', gridKNN.best_estimator_.p)
