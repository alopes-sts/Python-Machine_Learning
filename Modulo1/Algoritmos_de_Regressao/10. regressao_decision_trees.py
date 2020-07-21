# Exemplo de Algoritmo do Decision Tree para problemas de regressão
# O dataset é referente a possibilidades de aprovação de alunos

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

arquivo = pd.read_csv('C:/Onedrive/Curso_Machine_Learning/Modulo 1/Algoritmos_de_Regressao/datasets/Admission_Predict.csv')
arquivo.drop('Serial No.', axis=1, inplace=True)
y = arquivo['Chance of Admit ']
x = arquivo.drop('Chance of Admit ', axis=1)

# separando os dados em folds
kfold = KFold(n_splits=5, random_state=7)

# criação do modelo
modelo = DecisionTreeRegressor()
resultado = cross_val_score(modelo, x, y, cv=kfold)

# imprimindo o coef R2
# obs.: na documentação explica que o default deste modelo é R2
print('Coeficiente de determinação R2:', resultado.mean())
