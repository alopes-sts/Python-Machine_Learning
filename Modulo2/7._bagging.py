import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

pd.set_option('display.max_columns', 23)
arquivo = pd.read_csv('C:/Onedrive/Curso_Machine_Learning/Modulo 2/recipeData.csv', encoding="ISO-8859-1")
selecao = arquivo.loc[arquivo['StyleID'].isin([7, 10, 134, 9, 4, 30, 86, 12, 92, 6, 175, 39])]
selecao.drop('BeerID', axis=1, inplace=True)
selecao.drop('Name', axis=1, inplace=True)
selecao.drop('URL', axis=1, inplace=True)
selecao.drop('Style', axis=1, inplace=True)
selecao.drop('UserId', axis=1, inplace=True)
selecao.drop('PrimingMethod', axis=1, inplace=True)
selecao.drop('PrimingAmount', axis=1, inplace=True)

selecao['SugarScale'] = selecao['SugarScale'].replace('Specific Gravity', 0)
selecao['SugarScale'] = selecao['SugarScale'].replace('Plato', 1)

#transformando variáveis texto em colunas com o métdo one hot encoding
brewmethod_encode = pd.get_dummies(selecao['BrewMethod'])
selecao.drop('BrewMethod',axis=1, inplace=True)
#inserindo as variáveis cridas pelo one hot encoding no dataset
concatenado = pd.concat([selecao, brewmethod_encode], axis=1)
concatenado['PitchRate'].fillna(concatenado['PitchRate'].mean(), inplace=True)
concatenado.fillna(concatenado.median(), inplace=True)

#definindo variáveis preditoras e variável target
y = concatenado['StyleID']
x = concatenado.drop('StyleID', axis=1)

#criação do modelo para comparação com o bagging
modelo = DecisionTreeClassifier()
skfold = StratifiedKFold(n_splits=3)
resultado = cross_val_score(modelo, x, y, cv=skfold, n_jobs=-1)
print('Resultado do modelo: ', resultado.mean())  #o resultado será baixo

########## DEFININDO VALORES QUE SERÃO TESTADOS ##########
minimos_split = np.array([2, 3, 4, 5, 6, 7, 8])
maximo_nivel = np.array([5, 6, 7, 8, 9, 10, 11])
minimo_leaf = np.array([1, 2, 3, 4, 5, 6, 7, 8,])
valores_grid = {'min_samples_split': minimos_split, 'min_samples_leaf': minimo_leaf, 'max_depth': maximo_nivel}

#criação de outro modelo para comparação com o bagging
modelo = DecisionTreeClassifier()

#Criando os grids
gridDecisionTree = GridSearchCV(estimator=modelo, param_grid=valores_grid, cv=3, n_jobs=-1)
gridDecisionTree.fit(x, y)

#Imprimindo melhores resultados
print('\nTestando melhores parâmetros')
print('Minimo split: ', gridDecisionTree.best_estimator_.min_samples_split)
print('Maxima profundidade: ', gridDecisionTree.best_estimator_.max_depth)
print('Minimo leaf: ', gridDecisionTree.best_estimator_.min_samples_leaf)
print('Acurácia: ', gridDecisionTree.best_score_)

########## USANDO O BAGGING ##########
#Criação do modelo
#observação sobre o parâmetro max_samples, 0.5 significa 50%, se passarmos o valor 1, significará 1 variável e não 100%
#testar com valores diferentes de estimadores (n_estimators)
modelo = BaggingClassifier(n_estimators=1, max_samples=0.5, n_jobs=-1) #notar que são não for passado o parâmetro 'base_estimator', por default será usado decision tree
skfold = StratifiedKFold(n_splits=3)
resultado = cross_val_score(modelo, x, y, cv=skfold, n_jobs=-1)
print('\nResultado com o bagging: ', resultado.mean())

########## USANDO O BAGGING COM REGRESSÃO LOGISTICA ##########
#Criação do modelo
modelo = BaggingClassifier(base_estimator= LogisticRegression(), n_estimators=10, max_samples=0.5, n_jobs=-1)
skfold = StratifiedKFold(n_splits=3)
resultado = cross_val_score(modelo, x, y, cv= skfold, n_jobs=-1)
print('\nResultado do bagging com regressão logistica', resultado.mean())

########## USANDO O BAGGING COM NAIVE BAYES ##########
#Criação do modelo
modelo = BaggingClassifier(base_estimator=GaussianNB(), n_estimators=10, max_samples=0.5, n_jobs=-1)
skfold = StratifiedKFold(n_splits=3)
resultado = cross_val_score(modelo, x, y, cv= skfold, n_jobs=-1)
print('\nResultado usando Naive Bayes: ', resultado.mean())