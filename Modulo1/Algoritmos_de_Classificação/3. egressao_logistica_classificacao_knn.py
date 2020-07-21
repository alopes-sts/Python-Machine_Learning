# Exemplo de regressão logitica para classificação usando KNN
# O dataset faz parte da biblioteca do sklearn, referente a dados de cancer de mama
# notar que o dataset usado existem diversas colunas com ordens de grandeza
# diferentes (0.01, 1.1, 100,1, etc), por isso é necessário usar normalização
# knn sempre vai precisar de normalização dos dados

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', 30)
dados = load_breast_cancer()
x=pd.DataFrame(dados.data, columns=[dados.feature_names])
y=pd.Series(dados.target)

#print(x.head())

normalizador = MinMaxScaler(feature_range=(0, 1))
X_norm = normalizador.fit_transform(x)
X_treino, X_teste, Y_treino, Y_teste = train_test_split(X_norm, y, test_size=0.3)

# notar que é aqui que fica o valor de K (n_neighbors)
modelo = KNeighborsClassifier(n_neighbors=5)
modelo.fit(X_treino, Y_treino)

resultado = modelo.score(X_teste, Y_teste)
print('Acurácia:', resultado)