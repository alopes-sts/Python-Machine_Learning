from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
# obs.: foi escolhido GridSearchCV mas poderia ser RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.tree import export_graphviz
import pandas as pd
import numpy as np
import graphviz

# usando um dos datasets padrão do sklearn
iris = load_iris()
x = pd.DataFrame(iris.data, columns=[iris.feature_names])
y = pd.Series(iris.target)

# definindo os valores (dicionário) que serão testados em DecisionTree
minimos_split = np.array([2, 3, 4, 5, 6, 7, 8])
# testando os valores máximos para níveis da árvore, para saber qual a profundidade máxima (níveis) melhor se encaixa
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

# definindo variáveis para gerar o novo modelo, associando com os resultados acima
min_split2 = gridDecisionTree.best_estimator_.min_samples_split
max_profund = gridDecisionTree.best_estimator_.max_depth
alg_escolhido = gridDecisionTree.best_estimator_.criterion

# criando um arquivo para armazenamento da árvore
arquivo2 = 'C:/Onedrive/Curso_Machine_Learning/Modulo 1/Algoritmos_de_Classificação/datasets/exemplo_graphviz_decision_tree2.dot'
melhor_modelo = DecisionTreeClassifier(min_samples_split=(min_split2), max_depth=(max_profund), criterion=(alg_escolhido))
melhor_modelo.fit(x,y)

# gerando o gráfico da árvore de decisão
# modelo = é o modelo das dados após o treino (modelo.fit)
# arquivo = nome do arquivo de saída que foi criado ".dot"
# feature_names = é o nome das colunas do dataset, conferir na linha 9 onde declara a variável 'x'
export_graphviz(melhor_modelo, out_file=arquivo2, feature_names=iris.feature_names)
with open(arquivo2) as aberto:
    grafico_dot = aberto.read()
h = graphviz.Source(grafico_dot)
h.view()

