# Exemplos de tipos de scores que podem ser usados para problemas de classificação e regressão

import pandas as pd
arquivo = pd.read_csv('C:/Onedrive/Curso_Machine_Learning/Modulo 1/Algoritmos_de_Regressao/datasets/Admission_Predict.csv')
arquivo.drop('Serial No.', axis=1, inplace=True)
y = arquivo['Chance of Admit ']
x = arquivo.drop('Chance of Admit ', axis=1)

#############################
#####  primeiro exemplo #####
#############################
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
minimos_split = np.array([2, 3, 4, 5, 6, 7])
maximo_nivel = np.array([3, 4, 5, 6, 7, 9, 11])
algoritmo = ['mse', 'friedman_mse', 'mae']
valores_grid = {'min_samples_split': minimos_split, 'max_depth': maximo_nivel, 'criterion': algoritmo}
modelo = DecisionTreeRegressor()
# notar o parâmetro 'scoring'
gridDecisionTree = GridSearchCV(estimator=modelo, param_grid=valores_grid, cv=5, scoring='neg_mean_squared_error')
gridDecisionTree.fit(x,y)
print('Primeiro exemplo')
print('Minimo split: ', gridDecisionTree.best_estimator_.min_samples_split)
print('Máximo profundidade: ', gridDecisionTree.best_estimator_.max_depth)
print('Algoritmo escolhido: ', gridDecisionTree.best_estimator_.criterion)
print('Coeficiente R2: ', gridDecisionTree.best_score_)
print()
print()
print()


#############################
#####  segundo exemplo #####
#############################
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import median_absolute_error
kfold = KFold(n_splits=5)
modelo = DecisionTreeRegressor()
# notar o parâmetro 'scoring'
resultado = cross_val_score(modelo,x,y,cv=kfold,scoring='neg_mean_absolute_error')
print('Segundo Exemplo')
print('Erro médio absoluto:', resultado.mean())
print()
print()
print()


#############################
#####  terceiro exemplo #####
#############################
from sklearn.model_selection import train_test_split
# notar que neste exemplo está sendo importada a função median_absolute_error para usar o score
from sklearn.metrics import median_absolute_error
X_treino, X_teste, Y_treino, Y_teste = train_test_split(x, y, test_size=0.3)
modelo = DecisionTreeRegressor()
modelo.fit(X_treino,Y_treino)
predicoes = modelo.predict(X_teste)
erro=median_absolute_error(Y_teste, predicoes)
print('Terceiro exemplo')
print('Erro absoluto mediano: ', erro)







