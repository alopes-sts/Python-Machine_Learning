# Comparando modelos de regressão
# Linear / Ridge / Lasso / Elasticnet
# Download do dataset: https://www.kaggle.com/shivachandel/kc-house-data

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split

# cria função que gerar os resultados de treino e teste para cada um dos modelos de regressão
def modelosRegressao(x_treino, x_teste, y_treino, y_teste):
    regression = LinearRegression()
    ridge = Ridge()
    lasso = Lasso(alpha=100, max_iter=5000, tol=0.5)
    elastic_net = ElasticNet()
    regression.fit(x_treino, y_treino)
    ridge.fit(x_treino, y_treino)
    lasso.fit(x_treino, y_treino)
    elastic_net.fit(x_treino, y_treino)
    resultado_regression = regression.score(x_teste, y_teste)
    resultado_ridge = ridge.score(x_teste, y_teste)
    resultado_lasso = lasso.score(x_teste, y_teste)
    resultado_elastic_net = elastic_net.score(x_teste, y_teste)
    return imprime_resultados(resultado_regression, resultado_ridge, resultado_lasso, resultado_elastic_net)

# cria uma função que compara os resultados e diz qual o melhor
def imprime_resultados(resultado_regression, resultado_ridge, resultado_lasso, resultado_elastic_net):
    print("Resultado de Regressão Linear: ", resultado_regression,
          '\nResultado de Regressão Ridge: ', resultado_ridge,
          '\nResultado de Regressao Lasso: ', resultado_lasso,
          '\nResultado de Regressao Elastic Net: ', resultado_elastic_net)

    melhor = resultado_regression
    nome_melhor = 'Linear'
    if melhor < resultado_ridge:
        melhor = resultado_ridge
        nome_melhor = 'Ridge'
    if melhor < resultado_lasso:
        melhor = resultado_lasso
        nome_melhor = 'Lasso'
    if melhor < resultado_elastic_net:
        melhor = resultado_elastic_net
        nome_melhor = 'Elastic Net'

    print('\nO melhor resultado foi por regressão', nome_melhor, 'e o valor foi:', melhor)

# define o número de colunas que devem ser visualizadas, apesar de não ser necessário (apenas para efeitos didáticos)
# cria um dataframe a partir do arquivo .csv
# remove do dataframe as colunas não utilizadas
pd.set_option('display.max_columns', 21)
dataframe = pd.read_csv('C:/Onedrive/Curso_Machine_Learning/Modulo 1/Algoritmos_de_Regressao/datasets/kc_house_data.csv')
dataframe.drop(['id', 'date', 'zipcode', 'lat', 'long'], axis=1, inplace=True)

y = dataframe['price']  # define a coluna alvo (sempre deve ser na variável y da função)
x = dataframe.drop('price', axis=1)  # define a variável x como sendo o resto do dataframe

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.30, random_state=14)  # define o que será usado como teste e treino

modelosRegressao(x_treino, x_teste, y_treino, y_teste)  # faz a chamada da função para cálculo dos dados e exibe na tela os resultados





