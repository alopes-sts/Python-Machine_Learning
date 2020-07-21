import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import Ridge

pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 320)
arquivo = pd.read_csv('C:/Onedrive/Curso_Machine_Learning/Modulo 1/Algoritmos_de_Regressao/datasets/Admission_Predict.csv')
arquivo.drop('Serial No.', axis=1, inplace=True)
print(arquivo.head())

# separando as variáveis entre preditoras e variáveis target
y = arquivo['Chance of Admit ']
x = arquivo.drop['Chance of Admit ', axis=1]

# definindo o algoritmo de machine learning
modelo = Ridge()

# configurando o RFE
rfe = RFE(estimator=modelo, n_features_to_select=5)
fit = rfe.fit(x,y)

# mostrando os resultados
# .n_features = quantidade de features que foram selecionadas, no caso, são 5
# .support_ = vai mostrar quais atributos foram selecionados e quais não foram
# .ranking_ = mostra quais foram escolhidos (= 1) e mostra os não escolhidos, numerando, quanto maior o número, pior é
print('Número de atributos:', fit.n_features_)
print('Atributos selecionados:', fit.support_)
print('Ranking dos atributos:', fit.ranking_)

########################
# usando árvore de decisão ao inves do algoritmo de regressão (Ridge)
from sklearn.tree import DecisionTreeRegressor
y = arquivo['Chance of Admit ']
x = arquivo.drop('Chance od Admit ', axis = 1)

# definindo o algoritmo de machine learning
modelo = DecisionTreeRegressor()

# RFE
rfe = RFE(estimator=modelo, n_features_to_select=5)
fit = rfe.fit(x,y)

# exibindo os resultados
# obs.: notar que foram selecionadas features diferentes do Ridge
print('Número de atributos:', fit.n_features_)
print('Atributos selecionados:', fit.support_)
print('Ranking dos atributos:', fit.ranking_)
