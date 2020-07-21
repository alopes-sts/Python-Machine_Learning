# Regressão Linear - Testando diversos parâmetros com a função GridSearchCV
# Download do dataset: https://www.kaggle.com/adityadeshpande23/admissionpredictioncsv

import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV

dataframe = pd.read_csv('C:/Onedrive/Curso_Machine_Learning/Modulo 1/Algoritmos_de_Regressao/datasets/Admission_Predict.csv')
dataframe.drop('Serial No.', axis=1, inplace=True)
y = dataframe['Chance of Admit ']
x = dataframe.drop('Chance of Admit ', axis=1)

#cria dicionário para valores de alpha e l1_ratio
valores = {'alpha': [0.1, 0.5, 1, 2, 5, 10, 25, 50, 100],
           'l1_ratio': [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8]}

#define o modelo ElasticNet para uso
#faz a chamada da função GridSearchCV e passa os parâmetros necessários, usando o dicionário criado
modelo = ElasticNet()
procura = GridSearchCV(estimator=modelo, param_grid=valores, cv=5)
procura.fit(x,y)

#imprime os resultados usando as funções do GridSearchCV
print('Melhor score:', procura.best_score_)
print('Melhor alpha:', procura.best_estimator_.alpha)
print('Melhor l1_ratio:', procura.best_estimator_.l1_ratio)
