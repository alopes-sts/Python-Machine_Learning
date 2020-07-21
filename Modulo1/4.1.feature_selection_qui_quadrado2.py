import pandas as pd
from sklearn.datasets import load_iris
# função que verificar qual o melhor score do qui-quadrado
from sklearn.feature_selection import SelectKBest
# função do qui quadrado
from sklearn.feature_selection import chi2

# carregando o dataset
iris = load_iris()

# definindo as variáveis preditoras e a variável target
x = pd.DataFrame(iris.data, columns=[iris.feature_names])
y = pd.Series(iris.target)
print(x.head())
print()
print()

# selecionando duas variáveis com o maior qui-quadrado
# k=2 (as 2 melhores variáveis, ou seja as 2 melhores colunas)
algoritmo = SelectKBest(score_func=chi2, k=2)
dados_das_melhores_preditoras = algoritmo.fit_transform(x,y)

# resultados
print('Scores: ', algoritmo.scores_)
print('Resultado da transformação:\n', dados_das_melhores_preditoras)
