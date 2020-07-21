import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

pd.set_option('display.max_columns', 23)
arquivo = pd.read_csv('C:/Onedrive/Curso_Machine_Learning/Modulo 2/recipeData.csv', encoding='ISO-8859-1')

selecao = arquivo.loc[arquivo['StyleID'].isin([7, 10, 134, 9, 4, 30, 86, 12, 92, 6, 175, 39])]
selecao.drop(['BeerID', 'Name', 'URL', 'Style', 'UserId', 'PrimingMethod', 'PrimingAmount'], axis=1, inplace=True)
selecao['SugarScale'] = selecao['SugarScale'].replace('Specific Gravity', 0)
selecao['SugarScale'] = selecao['SugarScale'].replace('Plato', 1)

#usando one hot encoding para transformar variáveis texto em colunas
brewmethod_encode = pd.get_dummies(selecao['BrewMethod'])
#excluindo a coluna 'BrewMethod'
selecao.drop('BrewMethod', axis = 1, inplace=True)
#inserindo novamente as variáveis do one hot encoding no dataset
concatenado = pd.concat([selecao, brewmethod_encode], axis=1)

concatenado['PitchRate'].fillna(concatenado['PitchRate'].mean(), inplace=True)
concatenado.fillna(concatenado.median(), inplace=True)

y = concatenado['StyleID']
x = concatenado.drop('StyleID', axis=1)

#criação do modelo
modelo = AdaBoostClassifier()
skfold = StratifiedKFold(n_splits=3)
resultado = cross_val_score(modelo, x, y, cv=skfold, n_jobs=-1)
print(resultado.mean())

############## MELHORANDO OS PARÂMETROS ##############
valores_grid = {'learning_rate': np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])}

#criando modelo
modelo = AdaBoostClassifier(n_estimators=500)

#criando os grids
gridAdaBoost = GridSearchCV(estimator=modelo, param_grid=valores_grid, cv=3, n_jobs=-1)
gridAdaBoost.fit(x, y)

#imprimindo resultados
print('Melhor taxa de aprendizagem: ', gridAdaBoost.best_estimator_.learning_rate)
print('Acurácia: ', gridAdaBoost.best_score_)



############## TESTANDO NOVAMENTE COM OUTROS PARÂMETROS ##############
valores_grid = {'learning_rate': np.array([0.3, 0.3, 0.1, 0.05])}
#criando modelo
modelo = AdaBoostClassifier(n_estimators=500)
#criando os grids
gridAdaBoost = GridSearchCV(estimator=modelo, param_grid=valores_grid, cv=3, n_jobs=-1)
gridAdaBoost.fit(x, y)
#imprimindo resultados
print('\n\nMelhor taxa de aprendizagem: ', gridAdaBoost.best_estimator_.learning_rate)
print('Acurácia: ', gridAdaBoost.best_score_)


