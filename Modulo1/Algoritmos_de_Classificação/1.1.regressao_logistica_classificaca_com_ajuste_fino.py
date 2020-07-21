# Exemplo de Regressão Logistica para problemas de classificação com ajustes de parâmetros
# O dataset é referente a uma pesquisa sobre desodorantes (preço, fragância, etc)
# Download do dataset: https://www.kaggle.com/ramkumarr02/deodorant-instant-liking-data
# Execute no jupyter notebook pra melhor visualização

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

pd.set_option('display.max_columns', 64)
pd.set_option('display.max_rows', 64)
dataframe = pd.read_csv('C:/Onedrive/Curso_Machine_Learning/Modulo 1/Algoritmos_de_Classificação/datasets/Data_train_reduced.csv')

# comandos para verificar o dataset
print(dataframe.head())
print(dataframe.shape)
print(dataframe.dtypes)  # verificar que existem 1 coluna com dados não númericos (nome do produto), esta
# coluna deve ser removida, notar que existe uma coluna chamada código do produto que pode ser usada para identificação
# comandos para verificar dados faltantes
faltantes = dataframe.isnull().sum()
print(faltantes)
faltantes_percentual = (dataframe.isnull().sum() / len(dataframe['Product'])) * 100
print(faltantes_percentual)
# os dados faltantes devem ser tratados, iremos usar a mediana para os dados faltantes até 20% e o restante remover a coluna
# importante: a coluna 'q1_1.personal.opinion.of.this.Deodorant' foi removida pois ela é a opinião pessoal dos clientes
# na prática ela é a resposta do exercício, caso não seja retirada, o resultado será sempre de 100% de acurácia


dataframe.drop('q8.20', axis=1, inplace=True)
dataframe.drop('q8.18', axis=1, inplace=True)
dataframe.drop('q8.17', axis=1, inplace=True)
dataframe.drop('q8.8', axis=1, inplace=True)
dataframe.drop('q8.9', axis=1, inplace=True)
dataframe.drop('q8.10', axis=1, inplace=True)
dataframe.drop('q8.2', axis=1, inplace=True)
dataframe.drop('Respondent.ID', axis=1, inplace=True)
dataframe.drop('Product', axis=1, inplace=True)
dataframe.drop('q1_1.personal.opinion.of.this.Deodorant', axis=1, inplace=True)

# medianas das colunas faltantes
dataframe['q8.12'].fillna(dataframe['q8.12'].median(), inplace=True)
dataframe['q8.7'].fillna(dataframe['q8.7'].median(), inplace=True)

# seraparando as variáveis entre preditoras e variável target
y = dataframe['Instant.Liking']
x = dataframe.drop('Instant.Liking', axis=1)

# definindo valores para testes
valores_C = np.array([0.01, 0.1, 0.5, 1, 2, 3, 5, 10, 20, 50, 100])
regularizacao = ['l1', 'l2']
valores_grid = {'C': valores_C, 'penalty': regularizacao}

# criação do modelo
modelo = LogisticRegression()

# criando os grids
grid_regressao_logistica = GridSearchCV(estimator=modelo, param_grid=valores_grid, cv=5)
grid_regressao_logistica.fit(x,y)

print('Melhor acurácia: ', grid_regressao_logistica.best_score_)
print('Parâmetro C: ', grid_regressao_logistica.best_estimator_.C)
print('Regularização: ', grid_regressao_logistica.best_estimator_.penalty)



print ('Testando com stratifielkflod')
# separando os dados em folds
#stratifielkfold = StratifiedKFold(n_splits=5)

# criando o modelo
#modelo = LogisticRegression()
#resultado = cross_val_score(modelo,x,y,cv = stratifielkfold)
#print(resultado.mean())





