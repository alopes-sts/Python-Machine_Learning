# função que verificar qual o melhor score do qui-quadrado
from sklearn.feature_selection import SelectKBest
# função do qui quadrado
from sklearn.feature_selection import chi2

# definindo variáveis preditoras e target
x=[[12, 2, 30], [15, 11, 6], [16, 8, 90], [5, 3, 20],[4, 14, 5], [2, 5, 70]]
y=[1,1,1,0,0,0]

# selecionando duas variáveis com o maior qui-quadrado
# k=2 (as 2 melhores variáveis, ou seja as 2 melhores colunas)
algoritmo = SelectKBest(score_func=chi2, k=2)
dados_das_melhores_preditoras = algoritmo.fit_transform(x,y)

# resultados
print('Scores: ', algoritmo.scores_)
print('Resultado da transformação:\n', dados_das_melhores_preditoras)
