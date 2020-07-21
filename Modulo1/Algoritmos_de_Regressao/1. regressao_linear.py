# Exemplo de regressão linear, criando uma reta baseada em 200 pontos criados aleatóriamente
# make_regression cria dados aleatórios (gera um gráfico usando regressão linear)
# conforme os parametros informados (n_samples, n_features e noises)
# n_samples = quantidade de dados / n_features = quantidade de variaveis / noises = ruido
# a função make_regression cria um array com 200 números em "x" e "y"
# Fica melhor compreendido se executado no jupyter notebook em blocos separados

from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import numpy as np

# gerando uma massa de dados
x, y = make_regression(n_samples=200, n_features=1, noise=30)
# mostrando no gráfico
plt.scatter(x, y)
plt.show()

from sklearn.linear_model import LinearRegression
# criacao do modelo
modelo = LinearRegression()  # quando não passa nenhum parâmetro, está usando default:

modelo.fit(x,y)  # vai criar uma reta:
print('Coeficiente linear calculado:', modelo.intercept_)  # calcula coeficiente linear (y = mx + b), intercept é o "b":
print('Coeficiente angular calculado:', modelo.coef_)  # calcula coeficiente angular (y = mx + b), coef é o "m", é um array:

# mostrando o resultado
valor_intercept = modelo.intercept_  # associa o coeficiente linear a uma variavel:
valor_coef = modelo.coef_  # associa o coeficiente angular a uma variavel:
plt.scatter(x, y)
xreg = np.arange(-3, 3, 1)  # valores baseados no gráfico gerado pelo make_regression (-2 a 3, variando de 1 em 1):
plt.plot(xreg, valor_coef * xreg + valor_intercept, color='red')  # grafico regressao y=mx+b -> m=valor de .coef e b=valor de .intercept:
plt.show()

# funcao para criar os dados de teste e treino
from sklearn.model_selection import train_test_split  # a função 'train_test_split' divide os dados em conjunto de dados de treino e teste:
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size = 0.30)  # as variaveis X e Y irão pegar todos os pontos gerados por n_samples=200 do começo do código. teste_size irá pegar 30% dos dados para teste:
modelo.fit(x_treino, y_treino)
resultado = modelo.score(x_teste, y_teste)  # calcula o coeficiente R2 da reta criada com os dados de teste:
print('Coeficiente R2 da reta criada: ', resultado)

# visualizar a reta de teste que foi gerada
print('Coeficiente linear (reta de teste) calculado:', modelo.intercept_)
print('Coeficiente angular (reta de teste) calculado:', modelo.coef_)
valor_intercept = modelo.intercept_
valor_coef = modelo.coef_ 	
plt.scatter(x_teste, y_teste)
xreg = np.arange(-3, 3, 1)   
plt.plot(xreg, valor_coef * xreg+ valor_intercept, color='red')
plt.show()

# visualizar a reta de treino que foi gerada
print('Coeficiente linear (reta de teste) calculado:', modelo.intercept_)
print('Coeficiente angular (reta de treino) calculado:', modelo.coef_)
valor_intercept = modelo.intercept_
valor_coef = modelo.coef_ 	
plt.scatter(x_treino, y_treino)
xreg = np.arange(-3, 3, 1)   
plt.plot(xreg, valor_coef * xreg + valor_intercept, color='red')
plt.show()
