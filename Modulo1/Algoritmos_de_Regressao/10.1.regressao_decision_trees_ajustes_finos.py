# Exemplo de Algoritmo do Decision Tree para problemas de regressão com ajustes finos usandos um array para os dados que serão testados
# O dataset é referente a possibilidades de aprovação de alunos

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

arquivo = pd.read_csv('C:/Onedrive/Curso_Machine_Learning/Modulo 1/Algoritmos_de_Regressao/datasets/Admission_Predict.csv')
arquivo.drop('Serial No.', axis=1, inplace=True)
y = arquivo['Chance of Admit ']
x = arquivo.drop('Chance of Admit ', axis=1)

# definindo valores que serão testados em DecisionTree
minimos_split = np.array([2, 3, 4, 5, 6, 7])
maximo_nivel = np.array([3, 4, 5, 6, 7, 9, 11])
# lembrando, problema de regressão utiliza árvore de desvio padrão e cálculo da média
# o algoritmo abaixo é para escolher o melhor split entre os três
algoritmo = ['mse', 'friedman_mse', 'mae']
valores_grid = {'min_samples_split': minimos_split, 'max_depth': maximo_nivel, 'criterion': algoritmo}

# criação do modelo
modelo = DecisionTreeRegressor()

# criação dos grids
gridDecisionTree = GridSearchCV(estimator=modelo, param_grid=valores_grid, cv=5)
gridDecisionTree.fit(x, y)

# imprimindo os melhores parâmetros
print('Minimo split: ', gridDecisionTree.best_estimator_.min_samples_split)
print('Máximo profundidade: ', gridDecisionTree.best_estimator_.max_depth)
print('Algoritmo escolhido: ', gridDecisionTree.best_estimator_.criterion)
print('Coeficiente R2: ', gridDecisionTree.best_score_)

