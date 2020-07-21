# Exemplo regressão logistica para problemas de classificação usando Naive Bayes com GaussianNB
# Dataset par classificação de vinhos
# Download do dataset: https://www.kaggle.com/dell4010/wine-dataset

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB

pd.set_option('display.max_columns', 13)
dataframe = pd.read_csv('C:/Onedrive/Curso_Machine_Learning/Modulo 1/Algoritmos_de_Classificação/datasets/wine_dataset.csv')
# obs.: em programas anteriores foi necessário mudar os valores da coluna alvo, pois os dados dela são uma string
# no caso deste algoritmo isso não é necessário

y = dataframe['style']
x = dataframe.drop('style', axis=1)

# criação do modelo
modelo = GaussianNB()
# criando o algoritmo de validação cruzada conforme foi solicitado no exercicio
skfold = StratifiedKFold(n_splits=3)

# score
resultado = cross_val_score(modelo, x, y, cv=skfold)
print(resultado.mean())
