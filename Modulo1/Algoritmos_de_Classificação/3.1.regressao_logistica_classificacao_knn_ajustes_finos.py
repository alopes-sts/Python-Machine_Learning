# Exemplo de regressão logitica para classificação usando KNN + criando um array para os ajustes finos

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

pd.set_option('display.max_columns', 30)
dados = load_breast_cancer()
x=pd.DataFrame(dados.data, columns=[dados.feature_names])
y=pd.Series(dados.target)

# obs.: deve-se sempre fazer a normalização dos dados
normalizador = MinMaxScaler(feature_range=(0, 1))
X_norm = normalizador.fit_transform(x)

#definindo os valores que serão testados
valores_K = np.array([3, 5, 7, 9, 11])
calculo_distancia = ['minkowski','chebyshev']
valores_p = np.array([1, 2, 3, 4])
valores_grid = {'n_neighbors':valores_K, 'metric':calculo_distancia, 'p':valores_p}

# criando o modelo
modelo = KNeighborsClassifier()

# criando os grids
gridKNN = GridSearchCV(estimator = modelo, param_grid=valores_grid, cv=5)
gridKNN.fit(X_norm,y)

# imprimindo os melhores parâmetros
print('Melhor acurácia: ', gridKNN.best_score_)
print('Melhor K: ', gridKNN.best_estimator_.n_neighbors)
print('Método distância: ', gridKNN.best_estimator_.metric)
print('Melhor valor p: ', gridKNN.best_estimator_.p)

