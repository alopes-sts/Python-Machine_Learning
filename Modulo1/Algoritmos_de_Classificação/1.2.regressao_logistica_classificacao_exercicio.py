# Exemplo de regressão logistica para problemas de classificação
# no sklearn existem datasets de exemplos (consultar sklearn.datasets), iremos usar um sobre cancer de mama

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 30)
dados = load_breast_cancer()
# o dataset está em um formato específico do sklearn, por isso as variáveis estão sendo definidas desta forma
# o dataset do sklearn será convertido para um dataframe pandas
# as informações que estão no parâmetro 'columns' é padrão do sklearn, esta é a identificação das colunas (features_names)
x = pd.DataFrame(dados.data, columns=[dados.feature_names])
y = pd.Series(dados.target)  # no dataset do sklearn já existe uma coluna com a informação de 'target', basta associar com a variável y

# para visualizar os dados do dataset
#   x.head()
# verificar número de linhas e colunas
#   print(x.shape, y.shape)

# testar com arrays diferentes
valores_C = np.array([0.01, 0.1, 0.5, 1, 2, 3, 5, 10, 20, 50, 100])
regularizacao = ['l1', 'l2']
valores_grid = {'C':valores_C, 'penalty': regularizacao}

# definindo o modelo que será usado
modelo = LogisticRegression(C=1, penalty='l1', solver='liblinear', max_iter=6000)

# criando os grids
grid_regressao_logistica = GridSearchCV(estimator= modelo, param_grid= valores_grid, cv=5)
grid_regressao_logistica.fit(x,y)

print('Melhor acurácia: ', grid_regressao_logistica.best_score_)
print('Parâmetro C: ', grid_regressao_logistica.best_estimator_.C)
print('Regularização: ', grid_regressao_logistica.best_estimator_.penalty)
