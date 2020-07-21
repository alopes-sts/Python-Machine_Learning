import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import Ridge

pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 320)
arquivo = pd.read_csv('C:\Onedrive/Curso_Machine_Learning/Modulo 1/Algoritmos_de_Regressao/datasets/Admission_Predict.csv')
arquivo.drop('Serial No.', axis=1, inplace=True)
print(arquivo.head())

# separando as variaveis entre preditoras e variáavel target
y = arquivo['Chance of Admit ']
x = arquivo.drop('Chance of Admit ', axis=1)

# definindo o algoritmo de machine learning
modelo = Ridge()


