from sklearn.datasets import load_iris
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

# carregando o dataset
iris = load_iris()

# definindo as variáveis preditoras e a variável target
x = pd.DataFrame(iris.data, columns=[iris.feature_names])
y = pd.Series(iris.target)

print(x.head())

# selecionando 2 variáaveis com o maior f-value
algoritmo = SelectKBest(score_func=f_classif, k=2)
dados_das_melhores_preditoras = algoritmo.fit_transform(x,y)

# resultados
print('Scores: ', algoritmo.scores_)
print('Resultado da transformação:\n', dados_das_melhores_preditoras)