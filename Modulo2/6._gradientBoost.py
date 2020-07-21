import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

iris = load_iris()
x = pd.DataFrame(iris.data, columns=[iris.feature_names])
y = pd.Series(iris.target)

#print(x.head()) #verificando o dataset desconsiderando a variável target

#criação do modelo
modelo = GradientBoostingClassifier()
skfold = StratifiedKFold(n_splits=5)
resultado = cross_val_score(modelo, x, y, cv=skfold, n_jobs=-1)
print('Resultado foi: ', resultado.mean())

########## Ajustes finos ##########
minimos_split = np.array([2, 3, 4, 5, 6])
maximo_nivel = np.array([3, 4, 5, 6, 7])
minimo_leaf = np.array([2, 3, 4, 5, 6])
valores_grid = {'min_samples_split': minimos_split, 'max_depth': maximo_nivel, 'min_samples_leaf': minimo_leaf}

#Criação do modelo
modelo = GradientBoostingClassifier(n_estimators=250)

#Criando os grids
gridGB = GridSearchCV(estimator= modelo, param_grid= valores_grid, cv=3, n_jobs=-1)
gridGB.fit(x, y)

#Imprimindo os melhores parâmetros
print('\n\nTestando melhores parâmetros')
print('Minimo split: ', gridGB.best_estimator_.min_samples_split)
print('Máxima profundidade: ', gridGB.best_estimator_.max_depth)
print('Mínimo leaf: ', gridGB.best_estimator_.min_samples_leaf)
print('Acurácia: ', gridGB.best_score_)

