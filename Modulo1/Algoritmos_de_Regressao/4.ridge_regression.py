# Exemplo de Ridge Regression
# Download do dataset: https://www.kaggle.com/shivachandel/kc-house-data

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

pd.set_option('display.max_columns', 21)

# abrindo arquivo e removendo colunas indesejadas
arquivo = pd.read_csv('C:/Onedrive/Curso_Machine_Learning/Modulo 1/Algoritmos_de_Regressao/datasets/kc_house_data.csv')
arquivo = arquivo.drop(['id', 'date', 'zipcode', 'lat', 'long'], axis=1)

modelo = LinearRegression()  # associa a funcao de regressao linear a uma variavel

y = arquivo['price']  # cria um dataframe alvo com a coluna 'price', deve ser sempre a variável y
x = arquivo.drop('price', axis=1)  # cria um dataframe com toda a planilha exceto a coluna 'price'
# random_state define que os valores de testes serão sempre os mesmo todas as vezes que o programa for executado
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.30, random_state=14)

# aplica o modelo de regressao linear para as variáveis de teste e treino e verifica o resultado (coeficiente de determinacao)
modelo.fit(x_treino, y_treino)
resultado = modelo.score(x_teste, y_teste)
print('Resultado com Linear Regression:', resultado)

# aplica o modelo de regressão ridge, o parâmetro 'alpha=1' é default, pode ser retirado
# o valor de 'alpha' pode ser alterado para testes, o objetivo é verificar qual o valor
# que mais se adequa ao desejado, lembrando que o resultado deve ser o mais próximo de 1
modeloRidge = Ridge(alpha=1)
modeloRidge.fit(x_treino, y_treino)
resultadoRidge = modeloRidge.score(x_teste, y_teste)
print('Resultado com Rigde Regression:', resultadoRidge)
