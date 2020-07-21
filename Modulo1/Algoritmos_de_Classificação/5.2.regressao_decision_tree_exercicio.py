# Exemplo de problema de regressão logistica para problemas de classificação
# Dataset referente a dados de pacientes de ortopedia
# Download do dataset: https://www.kaggle.com/uciml/biomechanical-features-of-orthopedic-patients

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

arquivo = pd.read_csv('C:/Onedrive/Curso_Machine_Learning/Modulo 1/Algoritmos_de_Regressao/datasets/column_2C_weka.csv')
arquivo['class']=arquivo['class'].replace('Abnormal', 1)
arquivo['class']=arquivo['class'].replace('Normal', 0)
y = arquivo['class']
x = arquivo.drop('class', axis=1)

# definindo valores que serão testados em DecisionTree
minimos_split = np.array([2, 3, 4, 5, 6, 7, 8])
maximo_nivel = np.array([3, 4, 5, 6])

# testando qual o melhor algoritmo
algoritmo = ['gini','entropy']
valores_grid = {'min_samples_split': minimos_split, 'max_depth': maximo_nivel, 'criterion': algoritmo}

# criação do modelo
modelo = DecisionTreeClassifier()

# criando os grids
gridDecisionTree = GridSearchCV(estimator=modelo, param_grid=valores_grid, cv=5)
gridDecisionTree.fit(x,y)

# imprimindo os melhores parâmetros
print('Minimo split: ', gridDecisionTree.best_estimator_.min_samples_split)
print('Máximo profundidade: ', gridDecisionTree.best_estimator_.max_depth)
print('Algoritmo escolhido: ', gridDecisionTree.best_estimator_.criterion)
print('Acurácia: ', gridDecisionTree.best_score_)
