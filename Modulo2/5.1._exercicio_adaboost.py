#EXERCICIO DE TESTE COM O MODELO ADABOOST REGRESSOR
#dataset:  https://www.mldata.io/dataset-details/school_grades/

import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

pd.set_option('display.max_columns', 33)   #foi verficado com o comando arquivo.head() o número de colunas do dataset
arquivo = pd.read_csv('C:/Onedrive/Curso_Machine_Learning/Modulo 2/school_grades_weka_dataset.csv')

#verificando se existem dados faltantes
#faltantes = arquivo.isnull().sum()
#faltantes_percentual = (arquivo.isnull().sum() / len(arquivo['school'])) * 100
#print(faltantes_percentual)

#verificando o total de dados diferentes em cada coluna
#para decidir se usa o "one hot encode" ou se altera os dados para número
#print([arquivo[c].value_counts() for c in list(arquivo.columns)])

#usando o one hot encode
arquivo_encode = pd.get_dummies(arquivo, drop_first=False)

#conferindo como ficou o dataset (fica melhor visualizado no jupyter notebook)
#pd.set_option('display.max_columns', 59)     #com o comando "arquivo_encode.head()" pode-se verificar o número de colunas e linhas
#print(arquivo_encode.head())

#definindo variável target e variáveis preditoras, no site onde está o dataset está definido esta coluna como preditora
x = arquivo_encode.drop('G3', axis=1)
y = arquivo_encode['G3']

#criando o modelo
modelo = AdaBoostRegressor()
kfold = KFold(n_splits=5)
resultado = cross_val_score(modelo, x, y, cv=kfold, n_jobs=-1)
print('Resultado com kfold: ', resultado.mean())

#melhorias 1 - usando parâmetros no modelo
modelo = AdaBoostRegressor(n_estimators=500, learning_rate=0.1)  #notar que quando quanto maior os estimadores, menor deve ser o learning rate (verificar a teória sobre estes algoritmos
kfold = KFold(n_splits=5)
resultado = cross_val_score(modelo, x, y, cv=kfold, n_jobs=-1)
print('\nResultado com kfold alterando parâmetros: ', resultado.mean())

#melhorias 2 - substituindo kfold por stratifield kfold
modelo = AdaBoostRegressor(n_estimators=50)
skfold = StratifiedKFold(n_splits=5)
resultado = cross_val_score(modelo, x, y, cv=skfold, n_jobs=-1)
print('\nResultado com Stratified Kfold: ', resultado.mean())

#melhorias 3 - usando train test split
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.85, random_state=15)
modelo = AdaBoostRegressor(n_estimators=50)
skfold = StratifiedKFold(n_splits=3)
modelo.fit(x_treino, y_treino)
resultado = modelo.score(x_teste, y_teste)
print('\nResultado com Train Test Split: ', resultado)   #testar com valores diferentes em 'test_size'

#melhorias 4
valores_grid = {'learning_rate': np.array([0.3, 0.2, 0.1, 0.05])}
modelo = AdaBoostRegressor(n_estimators=500)
griAdaBoost = GridSearchCV(estimator= modelo, param_grid= valores_grid, cv= 5, n_jobs=-1)
griAdaBoost.fit(x, y)
print('\nTestando com GridSearchCV')
print('     Melhor taxa de aprendizagem: ', griAdaBoost.best_estimator_.learning_rate)
print('     Acurácia: ', griAdaBoost.best_score_)

