#analisando o dataset do naufrágio do Titanic usando o algoritmo do GradientBoost
#para verificar a probabilidade de sobrevivência de cada passageiro considerando idade, classe social, genero, etc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

pd.set_option('display.max_columns', 12)
arquivo = pd.read_csv('C:/Onedrive/Curso_Machine_Learning/Modulo 2/titanic_train.csv')
#verificando se existem dados faltantes, estes comandos só são executados na fase de preparação do dataset, pode-se usar o jupyter notebook para executa-los
#pd.set_option('display.max_rows', 159) # definindo o total de linhas para visualizar o dataset inteiro
#faltantes = arquivo.isnull().sum()
#faltantes_percentual = (arquivo.isnull().sum() / len(arquivo['PassengerId']))
#print(faltantes_percentual)
arquivo = arquivo.drop(['Ticket', 'Name', 'PassengerId', 'Cabin'], axis=1) #removendo colunas desnecessárias ou duplicadas, a coluna 'Cabin' foi removida pois tem muito dados faltantes
arquivo['Embarked'].replace(np.NaN, 'S', inplace = True) #substituindo os dados faltantes da coluna 'Embarked'

#existem dados faltantes na coluna 'Age', analisando em gráfico
#arquivo_encode.boxplot(column=['Age'])
#plt.show()
arquivo.fillna(arquivo['Age'].median(), inplace=True) #aplicando a mediana para os dados faltantes
#verificando novamente se os dados faltantes foram removidos
#faltantes = arquivo_encode.isnull().sum()
#faltantes_percentual = (arquivo_encode.isnull().sum() / len(arquivo_encode['PassengerId']))
#print(faltantes_percentual)

arquivo_encode = pd.get_dummies(arquivo, drop_first=False) #one hot encode, transformando colunas string em colunas com números

#definindo variáveis preditoras e variável target
x = arquivo_encode.drop('Survived', axis=1)
y = arquivo_encode['Survived']

#criação do modelo simples
modelo = GradientBoostingClassifier()
skfold = StratifiedKFold(n_splits=3)
resultado = cross_val_score(modelo, x, y, cv=skfold, n_jobs=-1)
print('O resultado foi: ', resultado.mean())

########## TESTANDO AJUSTES FINOS ##########
#criação do modelo alterando somente os valores de learning rate
valores_grid = {'learning_rate': [0.01, 0.04, 0.09, 0.6, 0.7]}
modelo = GradientBoostingClassifier(n_estimators=300) #na teoria, aumentando os estimators, o learning rate deveria ser mais baixo
gridGB = GridSearchCV(estimator= modelo, param_grid= valores_grid, cv=5, n_jobs=-1)
gridGB.fit(x, y)
print('\n\nTestando melhor learning rate')
print('Melhor learning rate: ', gridGB.best_estimator_.learning_rate)
print('Acurácia: ', gridGB.best_score_)

#criação do modelo alterando outros valores também
minimos_split = np.array([2, 3, 4, 5, 6])
maximo_nivel = np.array([3, 4, 5, 6, 7])
minimo_leaf = np.array([2, 3, 4, 5, 6])
learning_rates = np.array([0.1, 0.2, 0.4, 0.6, 0.7])
valores_grid = {'min_samples_split': minimos_split, 'max_depth': maximo_nivel, 'min_samples_leaf': minimo_leaf, 'learning_rate': learning_rates}
modelo = GradientBoostingClassifier(n_estimators=10)
gridGB = GridSearchCV(estimator= modelo, param_grid= valores_grid, cv=3, n_jobs=-1)
gridGB.fit(x, y)
print('\n\nTestando melhores parâmetros')
print('Minimo split: ', gridGB.best_estimator_.min_samples_split)
print('Máxima profundidade: ', gridGB.best_estimator_.max_depth)
print('Mínimo leaf: ', gridGB.best_estimator_.min_samples_leaf)
print('Melhor learning rate: ', gridGB.best_estimator_.learning_rate)
print('Acurácia: ', gridGB.best_score_)


