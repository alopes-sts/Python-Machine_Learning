# Exemplo regressão logistica para problemas de classificação usando Naive Bayes com GaussianNB
# Dataset referente a classificação de um tipo especifico de flor

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

iris = load_iris()
x = pd.DataFrame(iris.data, columns=[iris.feature_names])
y = pd.Series(iris.target)

# comando somente para visualização dos dados
# print(x.head())

# separando os dados entre treino e teste
X_treino, X_teste, Y_treino, Y_teste = train_test_split(x, y, test_size=0.3, random_state=67)

# criação do modelo
modelo = GaussianNB()
modelo.fit(X_treino, Y_treino)

# score
resultado = modelo.score(X_teste, Y_teste)
print('Acurácia:', resultado)




