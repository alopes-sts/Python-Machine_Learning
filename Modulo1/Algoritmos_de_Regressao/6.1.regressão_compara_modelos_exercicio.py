# Comparando modelos de regressão
# Linear / Ridge / Lasso / Elasticnet
# Download do dataset: https://www.kaggle.com/adityadeshpande23/admissionpredictioncsv

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split

def compara(x_treino, x_teste, y_treino, y_teste):
    regression = LinearRegression()
    ridge = Ridge()
    lasso = Lasso()
    elastic = ElasticNet()

    regression.fit(x_treino, y_treino)
    ridge.fit(x_treino, y_treino)
    lasso.fit(x_treino, y_treino)
    elastic.fit(x_treino, y_treino)

    resultado_regression = regression.score(x_teste,y_teste)
    resultado_ridge = ridge.score(x_teste,y_teste)
    resultado_lasso = ridge.score(x_teste,y_teste)
    resultado_elastic = elastic.score(x_teste,y_teste)

# cria um dicionário para verificar qual o maior valor
# o parâmetro 'key' significa que está sendo usado o valor da chave do dicionário
    dicionario_resultado = {'Regressão Linear':resultado_regression, 'Regressão Ridge':resultado_ridge, 'Regressão Lasso':resultado_lasso, 'Regressão ElasticNet':resultado_elastic}
    melhor_modelo = max(dicionario_resultado, key=dicionario_resultado.get)

    print('Resultado de Regressão Linear: ', resultado_regression,
          '\nResultado de Regressão Ridge: ', resultado_ridge,
          '\nResultado de Regressao Lasso: ', resultado_lasso,
          '\nResultado de Regressao Elastic Net: ', resultado_elastic)

    print("\nO melhor modelo foi:", melhor_modelo, "\ncom o valor: ", dicionario_resultado[melhor_modelo])

dataframe = pd.read_csv('C:/Onedrive/Curso_Machine_Learning/Modulo 1/Algoritmos_de_Regressao/datasets/Admission_Predict.csv')
dataframe.drop('Serial No.', axis=1, inplace=True)
y = dataframe['Chance of Admit ']
x = dataframe.drop('Chance of Admit ', axis=1)
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.30, random_state=14)

compara(x_treino, x_teste, y_treino, y_teste)

