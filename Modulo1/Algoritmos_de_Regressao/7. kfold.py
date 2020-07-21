# Exemplo de Regressão Linear com Validação Cruzada (Kfold)
# Download do dataset: https://www.kaggle.com/adityadeshpande23/admissionpredictioncsv

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

dataframe = pd.read_csv('C:/Onedrive/Curso_Machine_Learning/Modulo 1/Algoritmos_de_Regressao/datasets/Admission_Predict.csv')
dataframe.drop('Serial No.', axis=1, inplace=True)
y = dataframe['Chance of Admit ']
x = dataframe.drop('Chance of Admit ', axis=1)

modelo = LinearRegression()
kfold = KFold(n_splits=5)
resultado = cross_val_score(modelo, x, y, cv=kfold)

print('Resultado regressão linear com kfold:', resultado.mean())
