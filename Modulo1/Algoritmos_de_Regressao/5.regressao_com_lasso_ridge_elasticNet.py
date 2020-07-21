# Exemplo de comparação entre Elastic, Lasso e Ridge
# Download do dataset: https://www.kaggle.com/shivachandel/kc-house-data

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', 21)

# abrindo arquivo e removendo colunas indesejadas
arquivo = pd.read_csv('C:/Onedrive/Curso_Machine_Learning/Modulo 1/Algoritmos_de_Regressao/datasets/kc_house_data.csv')
arquivo = arquivo.drop(['id', 'date', 'zipcode', 'lat', 'long'], axis=1)

# associa a funcao de regressao linear a uma variavel
modelo = LinearRegression()

# cria um dataframe alvo com a coluna 'price', deve ser sempre a variável y
y = arquivo['price']
# cria um dataframe com toda a planilha exceto a coluna 'price'
x = arquivo.drop('price', axis = 1)
# random_state define que os valores de testes serão sempre os mesmo todas as vezes que o programa for executado
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.30, random_state=14)

# aplica o modelo de regressao linear para as variáveis de teste e treino e verifica o resultado (coeficiente de determinacao)
modelo.fit(x_treino, y_treino)
resultado = modelo.score(x_teste, y_teste)
print('Resultado com regressão linear:', resultado)

# aplica o modelo de regressão ridge, o parâmetro 'alpha=1' é default, pode ser retirado
# o valor de 'alpha' pode ser alterado para testes, o objetivo é verificar qual o valor
# que mais se adequa ao desejado, lembrando que o resultado deve ser o mais próximo de 1
modeloRidge = Ridge(alpha=10)
modeloRidge.fit(x_treino,y_treino)
resultadoRidge = modeloRidge.score(x_teste, y_teste)
print('Resultadom com regressão Ridge:', resultadoRidge)

# aplica o modelo de regressao lasso
# o valor de alpha foi ajustado paraa verificar qual valor tem melhor resultado
# valores baixos de alpha irão gerar um alerta quando for executado
# para isso, foi testado o parâmetro 'max_iter' (o padrão é 1000), porem continuou gerando alertas
# depois foi adicionado o parâmetro 'tol' (o padrão é 0.0001), este padrão verifica o resultado a
# cada iteração e se o resultado for menor que o valor de tol, significa que ele não precisa executar tantas iterações
# o valor padrão de tol é muito baixo, por isso pode ser necessário altera-lo, após isso, verificar o valor de max_iter
# pode reduzi-lo e verificar os resultados
modeloLasso = Lasso(alpha=100, max_iter=5000, tol=0.1)
modeloLasso.fit(x_treino,y_treino)
resultadoLasso = modeloLasso.score(x_teste, y_teste)
print('Resultado com regressão Lasso:', resultadoLasso)

# aplica o modelo de regressao Elastic Net
# o parâmetro 'l1_ratio' define a porcentagem de l1 e l2
modeloElasticNet = ElasticNet(alpha=1, l1_ratio=0.9, tol=0.2, max_iter=5000)
modeloElasticNet.fit(x_treino,y_treino)
resultadoElasticNet = modeloElasticNet.score(x_teste, y_teste)
print("Resultado com regressão Elastic:', resultadoElasticNet)
