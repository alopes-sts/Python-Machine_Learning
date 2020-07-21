# Exemplo de comparação de modelo de Regressão Linear com Validação Cruzada (Kfold) e os outros modelos (Ridge, Lasso e ElasticNet)
# Download do dataset: https://www.kaggle.com/adityadeshpande23/admissionpredictioncsv

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

def compara(x,y):
    regression = LinearRegression()
    ridge = Ridge()
    lasso = Lasso()
    elastic = ElasticNet()

# define o número de blocos que serão criados
    kfold = KFold(n_splits=10)

# executa as iterações do kfold em cada modelo de algoritmo
    result_regression = cross_val_score(regression, x, y, cv=kfold)
    result_ridge = cross_val_score(ridge, x, y, cv=kfold)
    result_lasso = cross_val_score(lasso, x, y, cv=kfold)
    result_elastic = cross_val_score(elastic, x, y, cv=kfold)

# cria um dicionário com a média de cada modelo de algoritmo
# verifica qual o melhor modelo
    dic_regmodels = {'Regressão Linear':result_regression.mean(), 'Regressão Ridge':result_ridge.mean(), 'Regressão Lasso':result_lasso.mean(), 'Regressão ElasticNet':result_elastic.mean()}
    melhor_modelo = max(dic_regmodels, key=dic_regmodels.get)

    print("\nO melhor modelo foi:", melhor_modelo, "\ncom o valor: ", dic_regmodels[melhor_modelo])

# abre o arquivo, remove a coluna 'Serial No.' que não é necessária e cria os dataframes
dataframe = pd.read_csv('C:/Onedrive/Curso_Machine_Learning/Modulo 1/Algoritmos_de_Regressao/datasets/Admission_Predict.csv')
dataframe.drop('Serial No.', axis=1, inplace=True)
y = dataframe['Chance of Admit ']
x = dataframe.drop('Chance of Admit ', axis=1)

# chama a função
compara(x,y)

