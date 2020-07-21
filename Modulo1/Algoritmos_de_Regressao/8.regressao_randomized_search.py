# Regressão Linear - Testando diversos parâmetros com a função Randomized Search
# Download do dataset: https://www.kaggle.com/adityadeshpande23/admissionpredictioncsv

import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import RandomizedSearchCV

dataframe = pd.read_csv('C:/Onedrive/Curso_Machine_Learning/Modulo 1/Algoritmos_de_Regressao/datasets/Admission_Predict.csv')
dataframe.drop('Serial No.', axis=1, inplace=True)
y = dataframe['Chance of Admit ']
x = dataframe.drop('Chance of Admit ', axis=1)

#cria um dicionário com valores de alpha e l1_ratio para serem usadas como parâmetros na função .radomizedSearchCV
valores = {'alpha': [0.1, 0.5, 1, 2, 5, 10, 25, 50, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000],
           'l1_ratio':[0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}

#define que o modelo de regressão é o Elastic Net
#faz a chamada da função .RandomizedSearchCV usando o dicionário acima como parâmetros
modelo = ElasticNet()
procura = RandomizedSearchCV(estimator=modelo, param_distributions=valores, n_iter=150, cv=5, random_state=15)
procura.fit(x, y)

# utiliza funções do .RandomizedSearchCV para exibir os melhores resultadoscap
print('Melhor score:', procura.best_score_)
print('Melhor alpha:', procura.best_estimator_.alpha)
print('Melhor l1_ratio:', procura.best_estimator_.l1_ratio)